import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import joblib
import requests
import time

# A helper function to classify cities into operational regions based on user definitions
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

def load_and_clean_data():
    columns = ["Time", "Location", "IsDrill", "id", "Threat", "Extra"]
    data = pd.read_csv("alarms_updated.csv", names=columns, skiprows=1, index_col=False)
    data["Time"] = pd.to_datetime(data["Time"], errors="coerce")
    data.dropna(subset=["Time", "Location"], inplace=True)
    return data

def train_regional_tactical_model():
    data = load_and_clean_data()
    
    # 1. Map Regions and Encode Locations
    data["Region"] = data["Location"].apply(get_region)
    encoder = LabelEncoder()
    data["Location_Encoded"] = encoder.fit_transform(data["Location"])

    # 2. Geographic Profiling
    loc_counts = data["Location_Encoded"].value_counts().to_dict()
    total_alerts = len(data)
    loc_risk_ratio = {k: v / total_alerts for k, v in loc_counts.items()}
    
    data["total_counts"] = data["Location_Encoded"].map(loc_counts)
    data["loc_risk_ratio"] = data["Location_Encoded"].map(loc_risk_ratio)
    data["is_rare"] = data["Location_Encoded"].apply(lambda x: 1 if loc_counts.get(x, 0) < 15 else 0)

    # 3. Time Series & Tactical Features
    data = data.sort_values(["Location_Encoded", "Time"])
    
    data["min_since_last"] = data.groupby("Location_Encoded")["Time"].diff().dt.total_seconds() / 60.0
    data["min_since_last"] = data["min_since_last"].fillna(-1)
    
    data["prev_gap"] = data.groupby("Location_Encoded")["min_since_last"].shift(1)
    data["acceleration"] = data["min_since_last"] - data["prev_gap"]
    data["acceleration"] = data["acceleration"].fillna(0)

    # --- THE NEW REGIONAL FEATURES ---
    data = data.sort_values("Time")
    data.set_index("Time", inplace=True)
    # Global Load
    data["country_load_1h"] = data["Location_Encoded"].rolling("1h").count().values
    data.reset_index(inplace=True)
    
    # Regional Load (How many alerts in this specific region in the last hour)
    data = data.sort_values(["Region", "Time"])
    data.set_index("Time", inplace=True)
    data["regional_load_1h"] = data.groupby("Region")["Location"].rolling("1h").count().reset_index(level=0, drop=True)
    data.reset_index(inplace=True)
    
    # Localized Impact combines region heat with specific city vulnerability
    data["localized_impact"] = data["regional_load_1h"] * data["loc_risk_ratio"]

    # Local 24h Load
    data = data.sort_values(["Location_Encoded", "Time"])
    data.set_index("Time", inplace=True)
    data["local_alerts_24h"] = data.groupby("Location_Encoded")["Location_Encoded"].rolling("24h").count().values
    data.reset_index(inplace=True)

    # Cyclical Time Features
    data["Hour_sin"] = np.sin(2 * np.pi * data["Time"].dt.hour / 24.0)
    data["Hour_cos"] = np.cos(2 * np.pi * data["Time"].dt.hour / 24.0)

    # 4. Target Formulation (Hard clipped to 12h max)
    data["next_alert"] = data.groupby("Location_Encoded")["Time"].shift(-1)
    data["Y_min_to_next"] = (data["next_alert"] - data["Time"]).dt.total_seconds() / 60.0
    
    data = data.dropna(subset=["Y_min_to_next"])
    train_set = data[data["Y_min_to_next"] <= 1440].copy()
    
    train_set["Y_min_clipped"] = train_set["Y_min_to_next"].clip(upper=720)
    train_set["Y_binary_6h"] = (train_set["Y_min_to_next"] <= 360).astype(int)

    # Added regional_load_1h to the features array
    features = [
        "Location_Encoded",# "Hour_sin", "Hour_cos", 
        "min_since_last", 
        "acceleration", "country_load_1h", "regional_load_1h", "localized_impact", 
        "local_alerts_24h", "is_rare", "total_counts"
    ]

    X = train_set[features]
    y_clf = train_set["Y_binary_6h"]
    y_reg = train_set["Y_min_clipped"]

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, shuffle=False 
    )

    # 5. Undersampling for Classifier
    train_clf_df = pd.concat([X_train, y_clf_train], axis=1)
    majority = train_clf_df[train_clf_df["Y_binary_6h"] == 1]
    minority = train_clf_df[train_clf_df["Y_binary_6h"] == 0]
    
    majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
    balanced_train = pd.concat([majority_downsampled, minority]).sample(frac=1, random_state=42)
    
    X_clf_train_bal = balanced_train[features]
    y_clf_train_bal = balanced_train["Y_binary_6h"]

    # 6. Train Models
    print("Training Pruned Regional Classifier...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
    clf.fit(X_clf_train_bal, y_clf_train_bal)

    print("Training Regional Regressor...")
    reg = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_reg_train)

    # 7. Final Training & Export
    full_df = pd.concat([X, y_clf], axis=1)
    maj_full = full_df[full_df["Y_binary_6h"] == 1]
    min_full = full_df[full_df["Y_binary_6h"] == 0]
    maj_down = resample(maj_full, replace=False, n_samples=len(min_full), random_state=42)
    bal_full = pd.concat([maj_down, min_full]).sample(frac=1, random_state=42)
    
    clf.fit(bal_full[features], bal_full["Y_binary_6h"])
    reg.fit(X, y_reg)

    last_alert_memory = data.groupby("Location_Encoded")["Time"].max().to_dict()
    last_gap_memory = data.groupby("Location_Encoded")["min_since_last"].last().to_dict()

    # model results and accuracy reports
    y_clf_pred = clf.predict(X_test)
    y_reg_pred = reg.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_clf_test, y_clf_pred))    
    print("\nMean Absolute Error for Regression:", mean_absolute_error(y_reg_test, y_reg_pred))
    print("\nFeature Importances:")
    for feat, imp in zip(features, clf.feature_importances_):
        print(f"{feat}: {imp:.4f}")

    joblib.dump(clf, "alert_classifier.joblib", compress=9)
    joblib.dump(reg, "alert_regressor.joblib", compress=9)
    joblib.dump(encoder, "location_encoder.joblib")
    joblib.dump({
        "time": last_alert_memory,
        "gap": last_gap_memory,
        "counts": loc_counts,
        "ratios": loc_risk_ratio
    }, "system_memory.joblib")
    joblib.dump(features, "feature_names.joblib")

    print("Success: Regional Engine v10.0 Saved.")

def find_all_unique_locations():
    data = load_and_clean_data()
    unique_locations = data["Location"].unique()
    print("Unique Locations in Dataset:")
    for loc in unique_locations:
        print(loc)


def update_data():
    starting_index = 85717
    df = pd.read_csv("alarms.csv", skiprows=starting_index)
    df.to_csv("alarms_updated.csv", mode="a", header=not pd.io.common.file_exists("alarms_updated.csv"), index=False)

if __name__ == "__main__":
    # Uncomment the function you want to run
    # update_data()
    # find_all_unique_locations()
    # classify_city_region("dummy")  # This will populate the region lists based on latitudes
    train_regional_tactical_model()

    