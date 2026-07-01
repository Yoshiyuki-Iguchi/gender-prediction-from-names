# Gender Prediction from First Names

> A simple tool that predicts gender from first names — and tells you when it's not sure.

🔗 **Live Demo:** https://gender-prediction-from-names-ns4ssvg2ke3uwma6sbs6de.streamlit.app/

## Demo ↓ (App may take a few seconds to wake up)

![demo](assets/demo.gif)

Note: This app is hosted on Streamlit Community Cloud, which may sleep after periods of inactivity. If you see a "sleep" screen, please wait a few seconds or click "Wake up app".

---

## How to Use

**1. Go to the link above**  
Just click the link — no installation, no login, nothing to download.

**2. Enter a name**  
Type any first name in the input box.

**3. Get results**  
You'll see:
- Predicted gender (Male / Female)
- Confidence score (0% to 100%)
- Category (Strong Male / Strong Female / Unisex)
- Warning if the name is ambiguous

**4. Try examples**  
Click any of the example buttons: James, Mary, Casey, Jordan, Riley, Quinn

---

## What This Tool Does

This is a machine learning model that looks at first names and predicts gender based on spelling patterns.

- Uses a dataset of ~100,000 names from US, UK, Canada, and Australia
- Analyzes character patterns: endings, first letters, vowel ratios, suffixes
- Gives a confidence score — **it tells you when it's not sure**

---

## Why Trust the Results?

| Category | Accuracy | Meaning |
|----------|----------|---------|
| Strong Female | 91.1% | Names like Mary, Elizabeth — clear female signal |
| Strong Male | 82.0% | Names like James, John — clear male signal |
| Unisex | 58.8% | Names like Casey, Jordan — truly ambiguous |

**Important:** The 58.8% accuracy for unisex names is **not a bug**.  
When a name is used by both genders, no model can reliably predict it.  
That's why the tool shows confidence scores and warnings — it's honest when it doesn't know.

---

## Try It Now

→ [https://gender-prediction-from-names-ns4ssvg2ke3uwma6sbs6de.streamlit.app/]

---

## Technical Overview (for the curious)

**Data:** 99,683 unique names from US Social Security, UK ONS, British Columbia, and Australian government data.

**Method:**
- Gaussian Mixture Model (GMM) to detect unisex names
- Random Forest Classifier (86.1% accuracy)
- Random Forest Regressor for confidence calibration

**Features:** 65 character-level features including first/last letters, suffixes, vowel ratio, and name length.

---

## License

MIT — free to use.
