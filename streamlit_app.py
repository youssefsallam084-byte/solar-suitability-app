# -*- coding: utf-8 -*-
"""
🌍 منصة تحليل الزحف العمراني واختيار مواقع الطاقة الشمسية
تطبيق ويب (Streamlit) — المستخدم بيشوف واجهة بس، من غير أي كود

طريقة التشغيل محليًا:
    pip install streamlit streamlit-folium earthengine-api geemap numpy matplotlib
    streamlit run streamlit_app.py

طريقة النشر على الإنترنت (مجانًا):
    1. ارفع الملف ده على GitHub repo
    2. روح https://share.streamlit.io وسجل دخول بحساب جوجل
    3. اختار الـ repo ودوس Deploy
    4. هيديك رابط عام أي حد يفتحه من المتصفح مباشرة
"""

import streamlit as st
import ee
import json
import base64
import numpy as np
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium
from folium.plugins import Draw

st.set_page_config(page_title="منصة تحليل الطاقة الشمسية", layout="wide", page_icon="🌍")

# ==============================================================================
# 1) واجهة الدخول: المستخدم يكتب اسم مشروعه بس، من غير ما يشوف أي كود
# ==============================================================================

if "ee_ready" not in st.session_state:
    st.session_state.ee_ready = False

if not st.session_state.ee_ready:
    st.title("🌍 منصة تحليل الزحف العمراني ومواقع الطاقة الشمسية")
    st.markdown("### 🔑 جاري الاتصال بـ Google Earth Engine تلقائيًا...")

    # ==========================================================================
    # الاتصال عبر Service Account (بيانات محفوظة في Streamlit Secrets)
    # ده بيشتغل تلقائيًا على السيرفر من غير أي نافذة تسجيل دخول أو انتظار
    # لازم تضيف بيانات الـ Service Account في: Settings > Secrets بصفحة التطبيق
    # ==========================================================================
    try:
        # قراءة الملف كامل كنص Base64 واحد (بدون أي مشاكل تنسيق TOML/newlines)
        b64_data = st.secrets["gee_service_account_b64"]
        json_str = base64.b64decode(b64_data).decode("utf-8")
        service_account_info = json.loads(json_str)
        project_id = service_account_info["project_id"]

        credentials = ee.ServiceAccountCredentials(
            service_account_info["client_email"],
            key_data=json_str
        )
        ee.Initialize(credentials, project=project_id)

        st.session_state.ee_ready = True
        st.session_state.project_id = project_id
        st.rerun()

    except KeyError:
        st.error(
            "❌ لسه معملتش إعداد الـ Service Account.\n\n"
            "روح لصفحة التطبيق على share.streamlit.io → ⋮ (النقط التلاتة) → Settings → Secrets، "
            "وضيف السطر: gee_service_account_b64 = \"...\""
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ فشل الاتصال بـ Google Earth Engine.\n\nتفاصيل الخطأ: {e}")
        st.stop()

# ==============================================================================
# دوال التحليل الأساسية (نفس منطق النسخة السابقة + إصلاح توقع 2030)
# ==============================================================================

@st.cache_data(show_spinner=False)
def _dummy_cache_key(project_id):
    return project_id


def get_training_data(_roi):
    area_sq_km = _roi.area(maxError=1).divide(1e6).getInfo()
    num_samples = max(20, min(400, int(area_sq_km / 2)))
    esa_lc = ee.Image('ESA/WorldCover/v200/2021').clip(_roi)
    remap_lc = esa_lc.select('Map').remap([50, 60, 30, 10, 20, 80], [0, 0, 1, 2, 2, 3], 1).rename('landcover')
    samples = remap_lc.addBands(ee.Image.pixelLonLat()).stratifiedSample(
        numPoints=num_samples, classBand='landcover', region=_roi, scale=30, geometries=True
    )
    return samples, remap_lc


def process_landsat_composite(year, _roi):
    target_year = year if year <= 2025 else 2025
    start_date, end_date = f'{target_year}-01-01', f'{target_year}-12-31'

    if target_year < 2013:
        collection = ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
        bands_in, thermal_in = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'], 'ST_B6'
    else:
        collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').merge(ee.ImageCollection('LANDSAT/LC09/C02/T1_L2'))
        bands_in, thermal_in = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'], 'ST_B10'

    filtered = collection.filterBounds(_roi).filterDate(start_date, end_date)
    if filtered.size().getInfo() == 0:
        raise ValueError(f"لا توجد مرئيات فضائية متاحة لهذه المنطقة في عام {target_year}.")

    median_img = filtered.median().clip(_roi)
    optical = median_img.select(bands_in).multiply(0.0000275).add(-0.2).rename(
        ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'])
    thermal = median_img.select(thermal_in).multiply(0.00341802).add(149.0).subtract(273.15).rename('LST_Celsius')

    image = optical.addBands(thermal)
    ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndbi = image.normalizedDifference(['SWIR1', 'NIR']).rename('NDBI')
    return image.addBands([ndvi, ndbi])


def calculate_advanced_suitability(_roi, landcover_img, year):
    dem = ee.Image('USGS/SRTMGL1_003').clip(_roi)
    slope = ee.Terrain.slope(dem)
    slopeSuit = slope.expression("(b(0) <= 3) ? 1.0 : (b(0) <= 6) ? 0.8 : (b(0) <= 10) ? 0.6 : 0.1", {'b(0)': slope})

    solar_year = year if year <= 2025 else 2025
    solar_coll = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR') \
        .select('surface_solar_radiation_downwards_sum') \
        .filterDate(f'{solar_year}-01-01', f'{solar_year}-12-31')
    solar = solar_coll.mean().clip(_roi)

    solar_min = solar.reduceRegion(ee.Reducer.min(), _roi, 1000).getNumber('surface_solar_radiation_downwards_sum')
    solar_max = solar.reduceRegion(ee.Reducer.max(), _roi, 1000).getNumber('surface_solar_radiation_downwards_sum')

    solarNorm = ee.Image.constant(0.5) if solar_min.eq(solar_max).getInfo() else solar.unitScale(solar_min, solar_max)

    landSuit = landcover_img.expression("(b(0) == 1) ? 1.0 : (b(0) == 0) ? 0.1 : 0.5", {'b(0)': landcover_img})
    suitability = slopeSuit.multiply(0.3).add(solarNorm.multiply(0.4)).add(landSuit.multiply(0.3)).rename('suitability_score')

    suitClass = suitability.expression(
        "(b(0) >= 0.75) ? 4 : (b(0) >= 0.55) ? 3 : (b(0) >= 0.35) ? 2 : 1", {'b(0)': suitability}
    ).rename('suitability_class')
    suitClass = suitClass.updateMask(ee.Image.constant(1).clip(_roi).mask())

    solar_kwh = solar.multiply(2.7778e-7)
    mean_solar = solar_kwh.reduceRegion(reducer=ee.Reducer.mean(), geometry=_roi, scale=1000).getInfo()

    class_hist = suitClass.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(), geometry=_roi, scale=100, maxPixels=1e10
    ).get('suitability_class').getInfo()

    return suitClass, suitability, mean_solar, class_hist, slope


# ------------------------------------------------------------------------------
# 🔧 الإصلاح الأساسي: توقع حقيقي للزحف العمراني للسنوات المستقبلية (زي 2030)
# بدل ما يرجع نفس رقم 2025 دايمًا، بنحسب اتجاه النمو من سنين سابقة فعلية
# ونستخدم الانحدار الخطي (Linear Regression) نتوقع بيه المساحة في المستقبل
# ------------------------------------------------------------------------------

def get_urban_area_km2(_roi, classified_img, year):
    """يرجع مساحة الزحف العمراني: حساب فعلي لو السنة <= 2025، أو توقع مبني على
    اتجاه نمو تاريخي حقيقي لو السنة في المستقبل (زي 2030)."""

    pixel_area = ee.Image.pixelArea().divide(1e6)

    def real_urban_km2_for_year(y):
        img = process_landsat_composite(y, _roi)
        # وكيل سريع لتقدير العمران بمؤشر NDBI (أسرع من إعادة تدريب RF لكل سنة مرجعية)
        urban_est = img.select('NDBI').gt(0.0)
        area = urban_est.multiply(pixel_area).reduceRegion(
            ee.Reducer.sum(), _roi, 100, maxPixels=1e10).getInfo().get('NDBI', 0)
        return area or 0

    # المساحة الفعلية الحالية (من تصنيف Random Forest الدقيق، أدق من الوكيل NDBI)
    current_urban = pixel_area.updateMask(classified_img.eq(0)).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=_roi, scale=30, maxPixels=1e10
    ).getInfo()
    current_km2 = current_urban.get('area', 0) if current_urban else 0

    if year <= 2025:
        return current_km2, None  # مفيش داعي لتوقع، ده رقم حقيقي فعلي

    # لو السنة مستقبلية: احسب اتجاه النمو من سنين سابقة فعلية وتوقع بيه المستقبل
    base_year = 2025
    ref_years = [y for y in [base_year - 20, base_year - 10, base_year - 5] if y >= 1990]
    points = []
    for y in ref_years:
        try:
            points.append((y, real_urban_km2_for_year(y)))
        except Exception:
            continue
    points.append((base_year, current_km2))

    if len(points) < 2:
        return current_km2, None  # مفيش بيانات كافية للتوقع، نرجع آخر رقم معروف

    years_arr = np.array([p[0] for p in points])
    area_arr = np.array([p[1] for p in points])
    slope_coef, intercept = np.polyfit(years_arr, area_arr, 1)
    slope_coef = max(slope_coef, 0)  # نفترض إن الزحف العمراني بيزيد مش بينقص

    predicted_km2 = slope_coef * year + intercept
    predicted_km2 = max(predicted_km2, current_km2)  # لازم يكون أكبر من أو يساوي الحالي

    trend_info = {
        "points": points,
        "growth_per_year_km2": round(slope_coef, 3),
        "predicted_year": year,
    }
    return predicted_km2, trend_info


# ==============================================================================
# 2) الواجهة الرئيسية بعد الاتصال
# ==============================================================================

st.title("🌍 منصة تحليل الزحف العمراني واختيار مواقع الطاقة الشمسية")
st.caption(f"متصل بمشروع: `{st.session_state.project_id}`")

col_controls, col_map = st.columns([1, 2.3])

with col_controls:
    st.subheader("⚙️ الإعدادات")
    year = st.slider("سنة التحليل:", min_value=1990, max_value=2030, value=2026, step=1)
    run_clicked = st.button("🚀 تشغيل التحليل", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("**📌 خطوات الاستخدام:**\n1. ارسم منطقة الدراسة على الخريطة (مربع أو مضلع)\n2. اختار السنة\n3. دوس تشغيل التحليل")

with col_map:
    st.subheader("🗺️ ارسم منطقة الدراسة")
    m = folium.Map(location=[26.8206, 30.8025], zoom_start=6, tiles="OpenStreetMap")
    Draw(export=False, draw_options={"rectangle": True, "polygon": True, "circle": False,
                                      "marker": False, "polyline": False, "circlemarker": False}).add_to(m)
    map_data = st_folium(m, width=750, height=450, key="draw_map")

# استخراج المنطقة اللي اترسمت
roi = None
if map_data and map_data.get("last_active_drawing"):
    geom = map_data["last_active_drawing"]["geometry"]
    roi = ee.Geometry(geom)

if run_clicked:
    if roi is None:
        st.error("❌ من فضلك ارسم منطقة الدراسة على الخريطة الأول.")
    else:
        with st.spinner("⏳ جاري التحليل... ممكن ياخد دقيقة أو اتنين حسب حجم المنطقة"):
            try:
                landsat_img = process_landsat_composite(year, roi)
                training_samples, base_lc = get_training_data(roi)

                class_bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'NDBI', 'LST_Celsius']
                training_data = landsat_img.select(class_bands).sampleRegions(
                    collection=training_samples, properties=['landcover'], scale=30, tileScale=16)
                classifier = ee.Classifier.smileRandomForest(50).train(training_data, 'landcover', class_bands)
                classified_img = landsat_img.select(class_bands).classify(classifier)

                urban_km2, trend_info = get_urban_area_km2(roi, classified_img, year)

                suitability_map, raw_suitability_img, solar_stats, class_hist, slope_img = \
                    calculate_advanced_suitability(roi, classified_img, year)
                solar_val = solar_stats.get('surface_solar_radiation_downwards_sum', 0) if solar_stats else 0

                # -------- عرض النتائج --------
                st.success("🎉 اكتمل التحليل بنجاح!")

                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("☀️ الإشعاع الشمسي", f"{solar_val:.2f} kWh/m²")
                mean_slope = slope_img.reduceRegion(ee.Reducer.mean(), roi, 1000).get('slope').getInfo()
                kpi2.metric("⛰️ متوسط الميل", f"{mean_slope:.2f}°")
                kpi3.metric("🏗️ الزحف العمراني", f"{urban_km2:.2f} كم²",
                            help="رقم متوقع (Projection) بناءً على اتجاه النمو التاريخي" if trend_info else "رقم فعلي محسوب")

                if trend_info:
                    st.info(
                        f"📈 هذا رقم **متوقع** لعام {year} (مش صورة قمر صناعي فعلية لأنها مش موجودة بعد)، "
                        f"مبني على معدل نمو تاريخي قدره **{trend_info['growth_per_year_km2']} كم²/سنة** "
                        f"محسوب من بيانات {', '.join(str(p[0]) for p in trend_info['points'])}."
                    )

                # -------- الداشبورد (رسوم بيانية) --------
                st.subheader("📊 Dashboard")
                hist = class_hist or {}
                labels_en = {'1': 'Unsuitable', '2': 'Poor', '3': 'Moderate', '4': 'Excellent'}
                colors = {'1': '#e74c3c', '2': '#f39c12', '3': '#f1c40f', '4': '#27ae60'}
                cats = ['1', '2', '3', '4']
                values = [hist.get(c, 0) for c in cats]
                total = sum(values) or 1
                percentages = [round(v / total * 100, 1) for v in values]

                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                axes[0].bar([labels_en[c] for c in cats], percentages, color=[colors[c] for c in cats])
                axes[0].set_title('Suitability Class Distribution (%)')
                for i, v in enumerate(percentages):
                    axes[0].text(i, v + 1, f'{v}%', ha='center', fontweight='bold')

                metrics = ['Solar\n(kWh/m2)', 'Slope\n(deg)', 'Urban\n(km2)']
                vals = [round(solar_val, 2), round(mean_slope, 2), round(urban_km2, 2)]
                axes[1].bar(metrics, vals, color=['#fbbf24', '#3498db', '#e74c3c'])
                axes[1].set_title('Key Indicators')
                for i, v in enumerate(vals):
                    axes[1].text(i, v, f'{v}', ha='center', va='bottom', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)

                # -------- مفتاح خريطة موحّد --------
                st.subheader("🔑 مفتاح الخريطة")
                legend_cols = st.columns(4)
                for i, c in enumerate(['4', '3', '2', '1']):
                    with legend_cols[i]:
                        st.markdown(
                            f"<div style='background:{colors[c]}; padding:8px; border-radius:6px; text-align:center; color:white;'>{labels_en[c]}</div>",
                            unsafe_allow_html=True)

                # -------- أفضل 10 مواقع --------
                sampled_points = raw_suitability_img.sample(
                    region=roi, scale=30, numPixels=500, geometries=True, seed=42, tileScale=16
                ).map(lambda f: f.set('score', f.get('suitability_score')))
                top_10 = sampled_points.sort('score', False).limit(10).getInfo()

                if top_10 and top_10.get('features'):
                    st.subheader("📍 أفضل 10 مواقع مقترحة")
                    rows = []
                    for i, feat in enumerate(top_10['features']):
                        lon, lat = feat['geometry']['coordinates']
                        score = feat['properties']['score']
                        rows.append({"الموقع": f"موقع {i+1}", "خط الطول": round(lon, 4),
                                     "خط العرض": round(lat, 4), "درجة الملاءمة": round(score, 2)})
                    st.dataframe(rows, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ حدث خطأ أثناء التحليل: {e}")

st.markdown("---")
if st.button("🔓 تسجيل خروج / تغيير المشروع"):
    st.session_state.ee_ready = False
    st.rerun()
