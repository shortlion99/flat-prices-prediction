# RAG Chatbot Evaluation Report

**Generated Time:** 2025-11-10 13:45:45  
**Data File:** ../data/hdb_rag/singapore_hdb_data.json

## Executive Summary

- **Ground Truth Accuracy:** 0.829
- **Response Quality:** 0.849
- **Price Prediction Handling:** 1.000
- **Response Time:** 0.800

**Overall RAG Performance:** 0.869

## Test Cases Used for Evaluation

Total test cases: **8**

**Area Info** (2 cases):
- Tell me about Clementi
- What are the pros of living in Tampines?

**Price Info** (1 cases):
- What's the price range for HDB flats in Yishun?

**Price Prediction** (2 cases):
- Can you predict HDB prices for next year?
- What will be the future price of flats in Punggol?

**Comparison** (1 cases):
- Compare living in Punggol vs Toa Payoh

**Unknown Area** (1 cases):
- Tell me about Marina Bay HDB housing options

**General** (1 cases):
- What is the weather like?

## Detailed Results

### Ground Truth Accuracy

**How it's derived:** Compares generated responses to manually created ground truth answers using semantic similarity (60%) combined with keyword matching (25%) and housing area mention accuracy (15%).

**Score:** 0.829

### Response Quality

**How it's derived:** Evaluates keyword presence (50%), area mention accuracy (30%), and response length appropriateness (20%).

**Score:** 0.849

**Sample Generated Answers:**

**Sample 1:**
- **Question:** What is the weather like?
- **Response:** I cannot provide real-time weather updates. However, you can check the current weather in Singapore using weather services like the **Meteorological Service Singapore (MSS)** or apps like Google Weather, AccuWeather, or Weather.com. Would you like me to guide you on how to check it?
- **Quality Score:** 0.625

**Sample 2:**
- **Question:** What will be the future price of flats in Punggol?
- **Response:** For price prediction and analytics, please refer to **our dashboard**, which has advanced machine learning models for HDB price forecasting and comprehensive market analytics.
- **Quality Score:** 1.000

### Price Prediction Handling

**How it's derived:** Measures proper redirection to Analytics Dashboard (70%) and avoidance of making actual predictions (30%).

**Score:** 1.000

**Sample Generated Answers for Price Prediction:**

**Sample 1:**
- **Question:** Can you predict HDB prices for next year?
- **Response:** For price prediction and analytics, please refer to **our dashboard**, which has advanced machine learning models for HDB price forecasting and comprehensive market analytics.
- **Redirect Score:** 1.000

**Sample 2:**
- **Question:** What will be the future price of flats in Punggol?
- **Response:** For price prediction and analytics, please refer to **our dashboard**, which has advanced machine learning models for HDB price forecasting and comprehensive market analytics.
- **Redirect Score:** 1.000

### Response Time

**How it's derived:** Measures average response time in seconds. Score: ≤2s = 1.0, ≤5s = 0.8, ≤10s = 0.6, >10s = 0.4.

**Score:** 0.800

**Average Response Time:** 2.58 seconds


## Comprehensive Test Cases and Results

| # | Question | Category | Generated Answer | Ground Truth Score | Response Quality Score | Overall Performance |
|---|----------|----------|------------------|-------------------|----------------------|-------------------|
| 1 | Tell me about Clementi | Area Info | Clementi is a mature town in Singapore known for its **excellent location near NUS and the one-north business hub**, with **strong connectivity via MRT (East-West Line) and expressways (AYE, PIE)**. HDB flats here range from **SGD 366K to 715K**, but many are older and in high demand, with **limited new BTO projects**. Nearby amenities include **Clementi Mall, Clementi Sports Hall, and West Coast Park**. | 0.924 | 1.000 | 0.962 |
| 2 | What are the pros of living in Tampines? | Area Info | Tampines is a mature town with full amenities and offers good MRT and bus connectivity. It is also conveniently close to Changi Airport. However, it can get crowded during peak hours. | 0.916 | 1.000 | 0.958 |
| 3 | What's the price range for HDB flats in Yishun? | Price Info | The price range for HDB flats in **Yishun** is **SGD 343,000 to 530,000**. It is considered relatively affordable compared to other areas in Singapore. | 0.951 | 1.000 | 0.976 |
| 4 | Can you predict HDB prices for next year? | Price Prediction | For price prediction and analytics, please refer to **our dashboard**, which has advanced machine learning models for HDB price forecasting and comprehensive market analytics. | 0.906 | 1.000 | 0.953 |
| 5 | What will be the future price of flats in Punggol? | Price Prediction | For price prediction and analytics, please refer to **our dashboard**, which has advanced machine learning models for HDB price forecasting and comprehensive market analytics. | 0.831 | 1.000 | 0.916 |
| 6 | Compare living in Punggol vs Toa Payoh | Comparison | Punggol offers a **modern, scenic waterfront lifestyle** with family-friendly parks and schools but has **fewer amenities** and **longer commutes** to the CBD. Toa Payoh is a **central, heritage-rich town** with **strong transport links** and amenities but features **older, smaller flats** and can feel **crowded**. Punggol’s flats are pricier (SGD 448K–620K) compared to Toa Payoh’s range (SGD 320K–776K). | 0.583 | 0.500 | 0.542 |
| 7 | Tell me about Marina Bay HDB housing options | Unknown Area | I don't know about Marina Bay HDB housing options as the provided context does not include this area. | 0.763 | 0.667 | 0.715 |
| 8 | What is the weather like? | General | I cannot provide real-time weather updates. However, you can check the current weather in Singapore using weather services like the **Meteorological Service Singapore (MSS)** or apps like Google Weather, AccuWeather, or Weather.com. Would you like me to guide you on how to check it? | 0.755 | 0.625 | 0.690 |

