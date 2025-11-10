# RAG Chatbot Evaluation Report

**Generated Time:** 2025-11-10 10:58:31  
**Data File:** ../data/hdb_rag/singapore_hdb_data.json

## Executive Summary

- **Ground Truth Accuracy:** 0.855
- **Response Quality:** 0.922
- **Price Prediction Handling:** 1.000
- **Response Time:** 0.800

**Overall RAG Performance:** 0.894

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
- Compare Woodlands and Queenstown in terms of housing

**Unknown Area** (1 cases):
- Tell me about Marina Bay housing options

**General** (1 cases):
- What is the weather like?

## Detailed Results

### Ground Truth Accuracy

**How it's derived:** Compares generated responses to manually created ground truth answers using semantic similarity (60%) combined with keyword matching (25%) and housing area mention accuracy (15%).

**Score:** 0.855

### Response Quality

**How it's derived:** Evaluates keyword presence (50%), area mention accuracy (30%), and response length appropriateness (20%).

**Score:** 0.922

**Sample Generated Answers:**

**Sample 1:**
- **Question:** What's the price range for HDB flats in Yishun?
- **Response:** The price range for HDB flats in **Yishun** is **SGD 343,000 to 530,000**. It is considered relatively affordable compared to other areas in Singapore.
- **Quality Score:** 1.000

**Sample 2:**
- **Question:** Compare Woodlands and Queenstown in terms of housing
- **Response:** I don't have information on Queenstown in the provided context, so I cannot compare it with Woodlands. Woodlands offers HDB flats priced between **SGD 362K–560K**, with pros like affordability and proximity to Malaysia, but it is far from the city center.
- **Quality Score:** 0.833

**Sample 3:**
- **Question:** Tell me about Clementi
- **Response:** Clementi is a mature town in Singapore known for its **excellent location near NUS and the one-north business hub**, with **strong connectivity via MRT and major expressways (AYE, PIE)**. HDB flats here range from **SGD 366K to 715K**, but many are older and in high demand, with limited new BTO projects. Nearby amenities include **Clementi Mall, Clementi Sports Hall, and West Coast Park**.
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

**Average Response Time:** 2.96 seconds

