import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import pytz
from datetime import datetime, timedelta

# UI Setup & Custom Styling
st.set_page_config(page_title="Iron Dome C2 System", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .main { background-color: #030305; color: #00f2ff; font-family: 'Orbitron', sans-serif; }
    .stMetric { background: rgba(0, 242, 255, 0.03); border: 1px solid rgba(0, 242, 255, 0.3); border-radius: 8px; padding: 20px; text-align: center; }
    .prediction-card {
        background: rgba(0, 242, 255, 0.02);
        padding: 30px; border-radius: 12px; border: 1px solid #00f2ff;
        box-shadow: 0 0 25px rgba(0, 242, 255, 0.1); margin-top: 25px;
    }
    .danger-card { border-color: #ff4b4b; box-shadow: 0 0 25px rgba(255, 75, 75, 0.1); }
    .safe-card { border-color: #00ff00; box-shadow: 0 0 25px rgba(0, 255, 0, 0.1); }
    </style>
    """, unsafe_allow_html=True)

# Helper function to classify region in live data
# Helper function for explicit regional mapping based on exact city lists
def get_region(city_name):
    city_str = str(city_name).strip()
    
    # Complete North region list
    north_cities = {
        "אביבים", "חולתה", "יסוד המעלה", "מרום גולן", "שעל", "אבירים", "אזור תעשייה כרמיאל", 
        "אזור תעשייה תרדיון", "איבטין", "אעבלין", "אשבל", "אשחר", "בית סוהר קישון", 
        "בית עלמין תל רגב", "דמיידה", "הר חלוץ", "חרשים", "יגור", "יובלים", "ימין אורד", 
        "יעד", "כאוכב אבו אלהיג'א", "כפר ח'וואלד", "כפר חסידים", "כרמיאל", "מורשת", "מנוף", 
        "מצפה אבי''ב", "מרכז אזורי משגב", "נחף", "ניר עציון", "סאג'ור", "עדי", "עין חוד", 
        "עספיא", "עצמון - שגב", "ערב אל נעים", "קורנית", "ראס עלי", "רכסים", "רמת יוחנן", 
        "רקפת", "שזור", "שכניה", "שפרעם", "אל עזי", "בית ג'אן", "חוסנייה", "יודפת", 
        "כמון - כמאנה מזרחית", "מכמנים - כמאנה מערבית", "מעלה צביה", "נופית", "סואעד חמירה", 
        "סכנין", "ראמה", "אמירים", "ביריה", "חזון", "חמדת ימים", "כפר חנניה", "כפר שמאי", 
        "לוטם וחמדון", "מורן", "מירון", "סלמה", "עין אל אסד", "עמוקה", "פרוד", "ראס אל-עין", 
        "שפר", "אזור תעשייה ציפורית", "בית שערים", "גבעת אלה", "דליה", "יקנעם המושבה והזורע", 
        "מנשית זבדה", "נהלל", "עין העמק", "עין השופט", "רמות מנשה", "רמת השופט", "שמשית", 
        "אלוני הבשן", "אבני איתן", "אזור תעשייה אלון התבור", "אזור תעשייה בני יהודה", 
        "אזור תעשייה קדמת גליל", "אילניה", "אלומות", "אלי עד", "אלמגור", "אמנון", "אניעם", 
        "אפיק", "אפיקים", "ארבל", "אשדות יעקב", "אתר ההנצחה גולני", "בית זרע", "בית ירח", 
        "בית קשת", "בני יהודה וגבעת יואב", "גבעת אבני", "גזית", "גינוסר", "גשור", "דבוריה", 
        "דברת", "דגניה א", "דגניה ב", "האון", "הודיות", "הזורעים", "ואדי אל חמאם", "חד נס", 
        "חמת גדר", "חספין", "טבריה", "יבנאל", "יונתן", "כדורי", "כינרת מושבה", "כינרת קבוצה", 
        "כנף", "כפר זיתים", "כפר חיטים", "כפר חרוב", "כפר כמא", "כפר מצר", "כפר נהר הירדן", 
        "כפר קיש", "כפר תבור", "כרכום", "לביא", "לבנים", "מבוא חמה", "מגדל", "מיצר", "מנחמיה", 
        "מסדה", "מעגן", "מעלה גמלא", "מצפה", "נאות גולן", "נבי שועייב", "נוב", "נטור", 
        "עין גב", "עין דור", "פוריה כפר עבודה", "פוריה נווה עובד", "פוריה עילית", "צמח", 
        "קשת", "רמות", "רמת מגשימים", "שדה אילן", "שדמות דבורה", "שיבלי אום אלג'נם", 
        "שער הגולן", "שרונה", "תל קציר", "אכסאל", "בית השיטה", "בלפוריה", "גבעת עוז", "גבת", 
        "גדעונה", "גן נר", "גניגר", "גשר", "דחי", "היוגב", "טורעאן", "טייבה בגלבוע", "יזרעאל", 
        "יפיע", "יפעת", "ישובי אומן", "ישובי יעל", "כפר ברוך", "כפר גדעון", "כפר החורש", 
        "כפר יחזקאל", "כפר כנא", "מגדל העמק", "מגן שאול", "מדרך עוז", "מולדת", "מוקיבלה", 
        "מזרע", "מרחביה מושב", "מרחביה קיבוץ", "משהד", "משמר העמק", "מתחם סקי גלבוע", 
        "נאעורה", "נווה אור", "נוף הגליל", "נורית", "נין", "נצרת", "סולם", "סנדלה", "עילוט", 
        "עין חרוד", "עין מאהל", "עפולה", "קבוצת גבע", "קיבוץ מגידו", "ריינה", "רם און", 
        "רמת דוד", "רמת צבי", "שריד", "תחנת רכבת כפר ברוך", "תל יוסף", "תל עדשים", "תמרת", 
        "רביד", "אזור תעשייה יקנעם עילית", "אליקים", "יקנעם עילית", "אורנים", "אלון הגליל", 
        "אלוני אבא", "אלונים", "בית אלפא וחפציבה", "בית יוסף", "בית לחם הגלילית", "בית שאן", 
        "בסמת טבעון", "גני חוגה", "דלית אל כרמל", "הושעיה", "הסוללים", "הרדוף", "זרזיר", 
        "חג'אג'רה", "חמדיה", "ירדנה", "כעביה טבאש", "כפר יהושע", "כפר תקווה", "כרם מהר''ל", 
        "מסילות", "ניר דוד", "עופר", "ציפורי", "קריית טבעון - בית זייד", "רמת ישי", "שדה יעקב", 
        "שדה נחום", "שער העמקים", "תחנת רכבת כפר יהושוע", "אבטליון", "בועיינה-נוג'ידאת", 
        "בית רימון", "מסד", "מצפה נטופה", "עוזייר", "עילבון", "רומאנה", "רומת אל הייב", 
        "חוקוק", "קצרין - אזור תעשייה", "אזור תעשייה צ.ח.ר", "איילת השחר", "אליפלט", "גדות", 
        "חצור הגלילית", "טובא זנגריה", "כורזים ורד הגליל", "כחל", "כפר הנשיא", "מחניים", 
        "מנחת מחניים", "מרכז אזורי רמת כורזים", "משמר הירדן", "עמיעד", "צפת - נוף כנרת", 
        "צפת - עיר", "צפת - עכברה", "קדמת צבי", "קצרין", "ראש פינה", "אורטל", "בית סוהר צלמון", 
        "דיר חנא", "הררית יחד", "טפחות", "כלנית", "מע'אר", "עין זיוון", "עראבה", "קדרים", 
        "חנתון", "ביר אלמכסור", "כפר מנדא", "מעוז חיים", "נוה איתן", "כפר גמילה מלכישוע", 
        "מירב", "מעלה גלבוע", "עין הנצי''ב", "רוויה", "רחוב", "רשפים", "שדה אליהו", 
        "שדי תרומות", "שלוחות", "שלפים", "תל תאומים", "חוף בצת", "ראש הנקרה", "שדמות מחולה", 
        "מחולה", "טירת צבי", "כפר רופין"
    }

    # Complete South region list
    south_cities = {
        "אזור תעשייה רבדים", "בית חלקיה", "חולדה", "יד בנימין", "יסודות", "כפר בן נון", 
        "כרמי יוסף", "נצר חזני", "רבדים", "אדורה", "אדורים", "אליאב", "בית חג\"י", "בני דקלים", 
        "היישוב היהודי חברון", "חוות אשכולות", "חוות מדבר חבר", "חוות מלאכי אברהם", 
        "כרמי קטיף ואמציה", "מעלה חבר", "מצפה זי\"ו", "נגוהות", "נטע", "עתניאל", "קרית ארבע", 
        "שומריה", "שקף", "אבנת", "איבי הנחל", "בית הברכה", "חוות נחלת אבות", "חוות תלם צפון", 
        "כפר אלדד", "כפר עציון", "כרמי צור", "מגדל עוז", "מיצד", "מעלה עמוס", "מעלה רחבעם", 
        "מצוקי דרגות", "מצפה שלם", "נוקדים", "נחושה", "עוז וגאון", "פארק תעשיות מגדל עוז", 
        "פני קדם", "צומת הגוש", "שדה בר", "תלם", "תקוע", "תקוע ד' וה'", "אורים", "ברוש", 
        "גבולות", "פטיש", "רנן", "תאשור", "תלמי אליהו", "אוהד", "צוחר", "שובל", "תדהר", 
        "אביגדור", "אורות", "אזור תעשייה באר טוביה", "אזור תעשייה גדרה", "אזור תעשייה הדרומי אשקלון", 
        "אזור תעשייה כנות", "אזור תעשייה עד הלום", "אזור תעשייה צפוני אשקלון", "אזור תעשייה תימורים", 
        "אחווה", "אלומה", "אמונים", "אשדוד - א,ב,ד,ה", "אשדוד - איזור תעשייה צפוני", "אשדוד - ג,ו,ז", 
        "אשדוד - ח,ט,י,יג,יד,טז", "אשדוד -יא,יב,טו,יז,מרינה,סיט", "אשקלון - דרום", "אשקלון - צפון", 
        "באר גנים", "באר טוביה", "ביצרון", "בית ניר", "בית עזרא", "בית שקמה", "בני דרום", 
        "בני עי''ש", "בני ראם", "ברכיה", "בת הדר", "גבעת וושינגטון", "גבעתי", "גדרה", "גיאה", 
        "גלאון", "גן הדרום", "גן יבנה", "גני טל", "גני יוחנן", "גת", "הודיה", "ורדון", "זבדיאל", 
        "זרחיה", "חפץ חיים", "חצב", "חצור", "יד נתן", "ינון", "כוכב מיכאל", "כנות", "כפר אביב", 
        "כפר אחים", "כפר הרי''ף וצומת ראם", "כפר ורבורג", "כפר מרדכי", "כפר סילבר", "כרם ביבנה", 
        "כרמי גת", "מבקיעים", "מזכרת בתיה", "מישר", "מנוחה", "מרכז שפירא", "משגב דב", "משואות יצחק", 
        "משען", "מתחם בני דרום", "נגבה", "נהורה", "נוגה", "נווה מבטח", "נחלה", "ניצן", "ניצנים", 
        "ניר בנים", "ניר גלים", "ניר ח''ן", "ניר ישראל", "סגולה", "עוצם", "עזר", "עזריקם", 
        "עין צורים", "ערוגות", "עשרת", "פארק תעשייה ראם", "קבוצת יבנה", "קדמה", "קדרון", "קוממיות", 
        "קריית גת", "קריית מלאכי", "רווחה", "שדה יואב", "שדה משה", "שדה עוזיהו", "שדמה", "שחר", 
        "שפיר", "שתולים", "תימורים", "תלמי יחיאל", "תלמי יפה", "מגן", "עין הבשור", "גבעולים", 
        "מלילות", "מעגלים", "שיבולים", "ישע", "מבטחים", "ניר עוז", "עמיעוז", "שדה ניצן", "שרשרת", 
        "אבשלום", "דקל", "חולית", "יבול", "יתד", "כרם שלום", "נווה", "ניר יצחק", "נירים", "סופה", 
        "עין השלושה", "פרי גן", "שדי אברהם", "שלומית", "תלמי יוסף", "אזור תעשייה נ.ע.מ", "אשבול", 
        "בית הגדי", "בית קמה", "זרועה", "יושיביה", "מבועים", "ניר משה", "ניר עקיבא", "נתיבות", 
        "פעמי תש''ז", "קלחים", "קסר א-סר", "שבי דרום", "שדה צבי", "תלמי ביל''ו", "תקומה", "בארי", 
        "חניון רעים אנדרטת הנובה", "כיסופים", "כפר מימון ותושיה", "רעים", "שוקדה", 
        "אזור תעשייה קריית גת", "לכיש", "נועם", "אבן שמואל", "זיקים", "חוף זיקים", "כרמיה", 
        "אחוזם", "איתן", "ברור חיל", "גברעם", "חלץ", "שדה דוד", "שלווה", "תלמים", "דורות", 
        "זמרת", "יכיני", "כפר עזה", "נחל עוז", "סעד", "עלומים", "שובה", "אור הנר", "איבים", 
        "ארז", "גבים", "יד מרדכי", "מטווח ניר עם", "מכללת ספיר", "מפלסים", "ניר עם", 
        "נתיב העשרה", "שדרות"
    }

    # Complete Negev region list
    negev_cities = {
        "אבו תלול", "אביגיל", "אום בטין", "אופקים", "אזור תעשייה מיתרים", "אזור תעשייה עידן הנגב", 
        "אל סייד", "אשכולות", "אשל הנשיא", "אשתמוע", "אתר דודאים", "באר שבע - דרום", "באר שבע - מזרח", 
        "באר שבע - מערב", "באר שבע - צפון", "בטחה", "בית יתיר", "גבעות בר", "גילת", "דביר", 
        "הר עמשא", "ואדי אל נעם דרום", "חוות דרומא", "חוות טואמין", "חוות טליה", "חוות יויו", 
        "חוות מור ואברהם", "חוות מנחם", "חוות מקנה יהודה", "חורה", "חירן", "חצרים", "טנא עומרים", 
        "כסייפה", "כרמים", "כרמית", "כרמל", "להב", "להבים", "לקיה", "מיתר", "מסלול", "מעון", 
        "מצפה יאיר", "מרעית", "משמר הנגב", "נבטים", "סוסיא", "סוסיא הקדומה", "סנסנה", "סעווה", 
        "עומר", "עשהאל", "פדויים", "צאלים", "קריית חינוך מרחבים", "רהט", "שגב שלום", "שמעה", 
        "שני ליבנה", "תארבין", "תל ערד", "תל שבע", "תפרח", "כפר הנוקדים", "מצפה מדרג", 
        "מרחצאות עין גדי", "עין גדי", "אבו קרינאת", "ערערה בנגב", "ביר הדאג'", "בני נצרים", 
        "מלונות ים המלח מרכז", "מצדה", "משאבי שדה", "עין בוקק", "רביבים", "רתמים", 
        "אזור תעשייה דימונה", "דימונה", "אשלים", "טללים", "ירוחם", "בתי מלון ים המלח", 
        "נווה זוהר", "אזור תעשייה רותם", "מצפה רמון", "עזוז", "שאנטי במדבר", "יהל", 
        "נאות סמדר", "נווה חריף", "שיטים", "לוטן", "קטורה", "אילת", "ספיר", "עין יהב", 
        "צופר", "צוקים", "רוחמה", "פארן", "באר מילכה"
    }

    # Absolute exact matching
    if city_str in north_cities:
        return "North"
    elif city_str in south_cities:
        return "South"
    elif city_str in negev_cities:
        return "Negev"
    else:
        # Defaults to Center for all remaining entries
        return "Center"

@st.cache_resource
def load_c2_assets():
    return (
        joblib.load("alert_classifier.joblib"),
        joblib.load("alert_regressor.joblib"),
        joblib.load("location_encoder.joblib"),
        joblib.load("system_memory.joblib"),
        joblib.load("feature_names.joblib")
    )

try:
    clf, reg, encoder, sys_mem, FEATURE_ORDER = load_c2_assets()
except Exception as e:
    st.error(f"System initialization failed. {e}")
    st.stop()

# --- THE NEW REGIONAL GITHUB INGESTION ENGINE ---
@st.cache_data(ttl=60)
def fetch_live_regional_load(target_region):
    csv_url = "https://raw.githubusercontent.com/yuval-harpaz/alarms/master/data/alarms.csv"
    try:
        df = pd.read_csv(csv_url, usecols=[0, 1], names=["Time", "Location"], skiprows=1)
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        
        israel_tz = pytz.timezone('Asia/Jerusalem')
        one_hour_ago = datetime.now(israel_tz).replace(tzinfo=None) - timedelta(hours=1)
        
        # Get alerts from the last 60 minutes
        recent_alerts = df[df["Time"] >= one_hour_ago].copy()
        
        country_total = len(recent_alerts)
        
        # Calculate how many of those were in our target region!
        recent_alerts["Region"] = recent_alerts["Location"].apply(get_region)
        regional_total = len(recent_alerts[recent_alerts["Region"] == target_region])
        
        return country_total, regional_total
        
    except Exception as e:
        print(f"Sync error with GitHub Data Stream: {e}")
        return 0, 0

# --- C2 COMMAND PANEL ---
with st.sidebar:
    st.header("🕹️ COMMAND PANEL")
    selected_city = st.selectbox("TARGET LOCATION", sorted(encoder.classes_))
    target_region = get_region(selected_city)
    
    israel_tz = pytz.timezone('Asia/Jerusalem')
    current_time = datetime.now(israel_tz).strftime("%Y-%m-%d %H:%M:%S")
    time_str = st.text_input("REFERENCE TIME", current_time)
    
    st.divider()
    
    sim_mode = st.toggle("ACTIVATE SIMULATION", False)
    if sim_mode:
        live_country_load = st.slider("SIM. NATIONAL LOAD", 0, 300, 80)
        live_regional_load = st.slider(f"SIM. REGIONAL LOAD ({target_region})", 0, 150, 20)
        local_load_24h_sim = st.slider("SIM. LOCAL 24H LOAD", 0, 50, 5)
        st.warning("SYSTEM IN SIMULATION MODE")
    else:
        live_country_load, live_regional_load = fetch_live_regional_load(target_region)
        local_load_24h_sim = 0 
        if live_country_load > 0:
            st.error(f"🔴 NATIONAL: {live_country_load} | REGIONAL: {live_regional_load}")
        else:
            st.success("🟢 LIVE STATUS: CLEAR NATIONWIDE")

    run_btn = st.button("EXECUTE ANALYSIS", use_container_width=True)

# --- MAIN DASHBOARD ENGINE ---
st.title("🛡️ IRON DOME C2 SYSTEM")
st.subheader("REGIONAL INTELLIGENCE ENGINE V10.0")

if run_btn:
    try:
        ref_time = pd.to_datetime(time_str)
        loc_idx = encoder.transform([selected_city])[0]
        
        loc_total = sys_mem['counts'].get(loc_idx, 0)
        risk_ratio = sys_mem['ratios'].get(loc_idx, 0.0001)
        is_rare = 1 if loc_total < 15 else 0
        
        last_t = sys_mem['time'].get(loc_idx)
        prev_gap = sys_mem['gap'].get(loc_idx, 0)
        
        min_since = (ref_time - last_t).total_seconds() / 60.0 if last_t else -1
        accel = min_since - prev_gap if min_since > 0 else 0
        
        # localized_impact uses the NEW regional load!
        local_impact = live_regional_load * risk_ratio
        
        if not sim_mode:
            local_alerts_24h = 1 if 0 < min_since <= 1440 else 0 
        else:
            local_alerts_24h = local_load_24h_sim

        input_dict = {
            "Location_Encoded": loc_idx,
            #"Hour_sin": np.sin(2 * np.pi * ref_time.hour / 24.0),
            #"Hour_cos": np.cos(2 * np.pi * ref_time.hour / 24.0),
            "min_since_last": float(min_since),
            "acceleration": float(accel),
            "country_load_1h": float(live_country_load),
            "regional_load_1h": float(live_regional_load),
            "localized_impact": float(local_impact),
            "local_alerts_24h": float(local_alerts_24h),
            "is_rare": float(is_rare),
            "total_counts": float(loc_total)
        }
        X_in = pd.DataFrame([input_dict])[FEATURE_ORDER]

        prob = clf.predict_proba(X_in)[0][1]
        tactical_time = reg.predict(X_in)[0]

        # --- SMART UI LOGIC & CONFIDENCE-BASED TWEAKS ---
        
        if prob < 0.40:
            is_safe = True
            display_time = "CLEAR"
        else:
            is_safe = False
            
            # --- THE 70% PIVOT TWEAK ---
            if prob < 0.70:
                tweak_multiplier = 1.0 + ((0.70 - prob + 0.1) * 10.0)  # Up to 10x increase as we approach 70% 
            else:
                tweak_multiplier = 1.0 - ((prob - 0.70) * 1.8) # Up to 1.8x decrease as we go beyond 70%
                
            adjusted_time = tactical_time * tweak_multiplier
            
            if adjusted_time < 2:
                adjusted_time = 2

            if loc_total < 10:
                adjusted_time *= 4  # Increase time window for very low historical counts
            
            if loc_total > 200:
                adjusted_time *= 0.8 # Decrease time window for very high historical count
                
            if adjusted_time >= 60:
                display_time = f"{int(adjusted_time/60)}h {int(adjusted_time%60)}m"
            else:
                display_time = f"{int(adjusted_time)} min"

        c1, c2, c3 = st.columns(3)
        c1.metric("6H THREAT PROBABILITY", f"{prob*100:.1f}%")
        c2.metric("ESTIMATED WINDOW", display_time)
        c3.metric(f"SECTOR: {target_region.upper()}", f"{loc_total} Hits")

        st.markdown("---")
        
        card_class = "safe-card" if is_safe else "danger-card"
        threat_status = "STABLE / CLEAR" if is_safe else "ELEVATED COMBAT READINESS"
        
        st.markdown(f"""
            <div class="prediction-card {card_class}">
                <h3>🔍 Tactical Intelligence Briefing: {selected_city}</h3>
                <p>Analysis completed for reference time: <b>{ref_time}</b>.</p>
                <ul>
                    <li><b>Operational Status:</b> {threat_status}</li>
                    <li><b>Sector Intelligence:</b> Model isolated the {target_region} command sector for hyper-accurate targeting.</li>
                    <li><b>Live Telemetry:</b> National Load: {live_country_load} | Sector Load: {live_regional_load}.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True) 

    except Exception as e:
        st.error(f"Tactical Engine Error: {e}")

st.divider()
st.caption("C2 PROTOCOL V10.0 | REGIONAL ISOLATION ENABLED | LIVE GITHUB DATALINK")