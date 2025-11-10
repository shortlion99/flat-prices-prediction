"""
RAG Evaluation Framework
Comprehensive evaluation suite for RAG chatbot performance including:
- Retrieval quality metrics
- Generation quality metrics  
- End-to-end evaluation
- Automated test case generation

"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import statistics
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_chatbot import RAGChatbot

@dataclass
class EvaluationResult:
    """Structure for storing evaluation results."""
    metric_name: str
    score: float
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class TestCase:
    """Structure for test cases with ground truth."""
    question: str
    ground_truth_answer: str 
    expected_areas: List[str]  # Singapore areas that should be mentioned in response
    expected_keywords: List[str]  # Keywords that should be in response
    category: str  # type of question
    difficulty: str  # "easy", "medium", "hard"

class RAGEvaluator:
    """Evaluator for RAG chatbot performance."""
    
    def __init__(self, data_file: str):
        """
        Initialize the evaluator.
        
        Args:
            data_file: Path to the data file used by RAG chatbot
        """
        self.data_file = data_file
        self.chatbot = None
        self.test_cases = self._create_test_cases()
        self.evaluation_results = []
        
        # Load available housing areas from data
        self.available_areas = self._load_available_areas()
        
    def _load_available_areas(self) -> List[str]:
        """Load available housing areas from the data file."""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [entry['area'] for entry in data]
        except Exception as e:
            print(f"Warning: Could not load areas from data file: {e}")
            return []
    
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases for evaluation."""
        test_cases = [
            TestCase(
                question="Tell me about Clementi",
                ground_truth_answer="Clementi is an excellent location near NUS and one-north business hub with strong connectivity via MRT and expressways. It's a mature town with established schools and amenities. However, flats are generally older and in high demand with limited space for new BTO projects.",
                expected_areas=["Clementi"],
                expected_keywords=["NUS", "connectivity", "mature"],
                category="area_info",
                difficulty="easy"
            ),
            TestCase(
                question="What are the pros of living in Tampines?",
                ground_truth_answer="The pros of living in Tampines include being a mature town with full amenities, good MRT and bus connectivity, and close proximity to Changi Airport.",
                expected_areas=["Tampines"],
                expected_keywords=["amenities", "MRT", "connectivity"],
                category="area_info",
                difficulty="easy"
            ),
            TestCase(
                question="What's the price range for HDB flats in Yishun?",
                ground_truth_answer="HDB flats in Yishun have a price range of SGD 343K - 530K, making it affordable compared to other areas.",
                expected_areas=["Yishun"],
                expected_keywords=["SGD", "price", "affordable"],
                category="price_info",
                difficulty="medium"
            ),
            
            # Price Prediction Questions - redirect to dashboard
            TestCase(
                question="Can you predict HDB prices for next year?",
                expected_areas=[],
                expected_keywords=["dashboard", "prediction", "analytics", "machine learning"],
                category="price_prediction",
                difficulty="easy",
                ground_truth_answer="For price predictions and forecasting, I'd recommend using the Analytics Dashboard which has machine learning models specifically designed for predicting HDB prices. You can find detailed price trend analysis and predictive models there that can give you insights into future price movements."
            ),
            TestCase(
                question="What will be the future price of flats in Punggol?",
                expected_areas=[],
                expected_keywords=["dashboard", "prediction", "analytics"],
                category="price_prediction", 
                difficulty="medium",
                ground_truth_answer="For specific price predictions for Punggol flats, please check the Analytics Dashboard where you'll find machine learning models that can forecast prices for different areas including Punggol. The dashboard provides detailed predictive analytics and price trend forecasts for various HDB locations."
            ),
            
            # Comparison Questions - Hard
            TestCase(
                question="Compare Woodlands and Ang Mo Kio in terms of housing",
                expected_areas=["Woodlands", "Queenstown"],
                expected_keywords=["price", "pros", "cons"],
                category="comparison",
                difficulty="hard",
                ground_truth_answer="Woodlands vs Queenstown Housing Comparison:\n\nWoodlands:\n- Generally more affordable HDB prices\n- Newer developments and town planning\n- Good connectivity to Malaysia via Causeway\n- More family-oriented with parks and recreational facilities\n- Typically 3-room flats range from $250k-350k, 4-room from $350k-450k\n\nQueenstown:\n- Higher prices due to central location\n- Mature estate with established amenities\n- Close to CBD and business districts\n- Mix of older and renovated flats\n- Typically 3-room flats range from $400k-550k, 4-room from $500k-700k\n\nQueenstown offers better location and connectivity but at higher prices, while Woodlands provides more affordable options with newer infrastructure."
            ),
            
            # Areas without HDB housing questions 
            TestCase(
                question="Tell me about Marina Bay housing options",
                expected_areas=[],
                expected_keywords=["don't know", "no information", "not available"],
                category="unknown_area",
                difficulty="medium",
                ground_truth_answer="I don't have specific information about HDB housing in Marina Bay. Marina Bay is primarily a commercial and business district with mostly private condominiums and commercial developments rather than HDB flats. For HDB housing information, I can help you with other residential areas that have HDB developments. Would you like me to suggest some nearby areas with HDB options?"
            ),
            
            # General Questions
            TestCase(
                question="What is your name?",
                expected_areas=[],
                expected_keywords=["assistant", "help", "Singapore", "housing"],
                category="general",
                difficulty="easy",
                ground_truth_answer="I'm an AI assistant specialized in helping with Singapore HDB housing information. I can provide details about different areas, housing prices, amenities, and help you find information about HDB flats across Singapore. How can I help you with your housing questions today?"
            ),
        ]
        
        return test_cases
    
    def initialize_chatbot(self):
        """Initialize the RAG chatbot for evaluation."""
        if self.chatbot is None:
            print("Initializing RAG chatbot for evaluation...")
            self.chatbot = RAGChatbot(self.data_file)
            print("Chatbot initialized successfully")
    
    # Evaluation Functions (retrieval-based questions) ======================================================
    
    def evaluate_response_quality(self) -> EvaluationResult:
        """
        Evaluate the quality of generated responses.
        """
        print("\nEvaluating response quality...")
        
        self.initialize_chatbot()
        quality_scores = []
        detailed_results = []
        
        for test_case in self.test_cases:
            print(f"  Testing: {test_case.question[:50]}...")
            
            # Get response from chatbot
            response = self.chatbot.chat(test_case.question)
            
            # Calculate quality metrics
            keyword_score = self._calculate_keyword_presence(response, test_case.expected_keywords)
            area_score = self._calculate_area_mention(response, test_case.expected_areas)
            length_score = self._calculate_response_length_score(response)
            
            # Overall quality score (weighted)
            overall_score = (keyword_score * 0.5 + area_score * 0.3 + length_score * 0.2)
            quality_scores.append(overall_score)
            
            detailed_results.append({
                "question": test_case.question,
                "response": response,
                "category": test_case.category,
                "difficulty": test_case.difficulty,
                "keyword_score": keyword_score,
                "area_score": area_score,
                "length_score": length_score,
                "overall_score": overall_score
            })
        
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0.0
        
        return EvaluationResult(
            metric_name="response_quality",
            score=avg_quality,
            details={
                "average_quality": avg_quality,
                "individual_scores": quality_scores,
                "detailed_results": detailed_results,
                "total_responses": len(quality_scores)
            },
            timestamp=datetime.now()
        )
    
    def _calculate_keyword_presence(self, response: str, expected_keywords: List[str]) -> float:
        """Calculate how many expected keywords are present in response."""
        if not expected_keywords:
            return 1.0
        
        response_lower = response.lower()
        matches = sum(1 for keyword in expected_keywords if keyword.lower() in response_lower)
        
        return matches / len(expected_keywords)
    
    def _calculate_area_mention(self, response: str, expected_areas: List[str]) -> float:
        """Calculate if expected areas are mentioned in response."""
        if not expected_areas:
            return 1.0
        
        response_lower = response.lower()
        matches = sum(1 for area in expected_areas if area.lower() in response_lower)
        
        return matches / len(expected_areas)
    
    def _calculate_response_length_score(self, response: str) -> float:
        """Score response based on appropriate word count."""
        word_count = len(response.split())
        
        if 10 <= word_count <= 100:  # Ideal range
            return 1.0
        elif 5 <= word_count < 10 or 100 < word_count <= 150:  # Acceptable range
            return 0.7
        elif word_count < 5:  # Too short
            return 0.3
        else:  # Too long
            return 0.5
    
    def evaluate_ground_truth_accuracy(self) -> EvaluationResult:
        """
        Evaluate response accuracy against ground truth answers using semantic similarity.
        """
        print("\nEvaluating ground truth accuracy...")
        
        self.initialize_chatbot()
        
        # Initialize sentence transformer for semantic similarity
        similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        accuracy_scores = []
        detailed_results = []
        
        for test_case in self.test_cases:
            print(f"  Testing: {test_case.question[:50]}...")
            
            # Get response from chatbot
            response = self.chatbot.chat(test_case.question)
            
            # Calculate semantic similarity with ground truth
            response_embedding = similarity_model.encode([response])
            ground_truth_embedding = similarity_model.encode([test_case.ground_truth_answer])
            
            # Caculate Cosine similarity
            similarity = float(np.dot(response_embedding[0], ground_truth_embedding[0]) / (
                np.linalg.norm(response_embedding[0]) * np.linalg.norm(ground_truth_embedding[0])
            ))
            
            # Additional scoring factors
            keyword_score = self._calculate_keyword_presence(response, test_case.expected_keywords)
            area_score = self._calculate_area_mention(response, test_case.expected_areas)
            
            # Combined accuracy score (semantic similarity weighted higher)
            combined_score = similarity * 0.6 + keyword_score * 0.25 + area_score * 0.15
            accuracy_scores.append(combined_score)
            
            detailed_results.append({
                "question": test_case.question,
                "generated_response": response,
                "ground_truth": test_case.ground_truth_answer,
                "semantic_similarity": similarity,
                "keyword_score": keyword_score,
                "area_score": area_score,
                "combined_score": combined_score,
                "category": test_case.category,
                "difficulty": test_case.difficulty
            })
        
        avg_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0.0
        
        return EvaluationResult(
            metric_name="ground_truth_accuracy",
            score=avg_accuracy,
            details={
                "average_accuracy": avg_accuracy,
                "individual_scores": accuracy_scores,
                "detailed_results": detailed_results,
                "total_comparisons": len(accuracy_scores)
            },
            timestamp=datetime.now()
        )

    # Evaluation Functions (Non-retrieval based questions & response time) ==================================

    def evaluate_price_prediction_handling(self) -> EvaluationResult:
        """
        Evaluate how well the system handles price prediction queries.
        """
        print("\nEvaluating price prediction handling...")
        
        self.initialize_chatbot()
        
        price_prediction_cases = [tc for tc in self.test_cases if tc.category == "price_prediction"]
        scores = []
        detailed_results = []
        
        for test_case in price_prediction_cases:
            response = self.chatbot.chat(test_case.question)
            
            # Check if response properly redirects to dashboard
            redirect_keywords = ["dashboard", "prediction", "analytics", "machine learning", "forecasting"]
            redirect_score = sum(1 for kw in redirect_keywords if kw.lower() in response.lower()) / len(redirect_keywords)
            
            # Ensure it doesn't try to make actual predictions
            avoid_keywords = ["will be", "predicted price", "future price is", "estimate is"]
            avoid_score = 1.0 - (sum(1 for kw in avoid_keywords if kw.lower() in response.lower()) / len(avoid_keywords))
            
            overall_score = (redirect_score * 0.7 + avoid_score * 0.3)
            scores.append(overall_score)
            
            detailed_results.append({
                "question": test_case.question,
                "response": response,
                "redirect_score": redirect_score,
                "avoid_score": avoid_score,
                "overall_score": overall_score
            })
        
        avg_score = statistics.mean(scores) if scores else 0.0
        
        return EvaluationResult(
            metric_name="price_prediction_handling",
            score=avg_score,
            details={
                "average_score": avg_score,
                "individual_scores": scores,
                "detailed_results": detailed_results,
                "total_queries": len(scores)
            },
            timestamp=datetime.now()
        )
    
    def evaluate_response_time(self) -> EvaluationResult:
        """
        Evaluate response time performance.
        """
        print("\nEvaluating response time...")
        
        self.initialize_chatbot()
        response_times = []
        detailed_results = []
        
        # Test with a subset of questions to avoid too long evaluation
        test_subset = self.test_cases[:5]  
        
        for test_case in test_subset:
            start_time = time.time()
            response = self.chatbot.chat(test_case.question)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            detailed_results.append({
                "question": test_case.question,
                "response_time": response_time,
                "category": test_case.category
            })
        
        avg_time = statistics.mean(response_times) if response_times else 0.0
        
        # Score based on response time (lower is better)
        if avg_time <= 2.0:
            time_score = 1.0
        elif avg_time <= 5.0:
            time_score = 0.8
        elif avg_time <= 10.0:
            time_score = 0.6
        else:
            time_score = 0.4
        
        return EvaluationResult(
            metric_name="response_time",
            score=time_score,
            details={
                "average_time": avg_time,
                "individual_times": response_times,
                "detailed_results": detailed_results,
                "total_queries": len(response_times)
            },
            timestamp=datetime.now()
        )
    
    # Running Evaluation ========================================================================================

    def evaluate_full_pipeline(self) -> Dict[str, EvaluationResult]:
        """
        Run comprehensive evaluation of the entire RAG pipeline.
        """
        print("Starting comprehensive RAG evaluation...")
        print("=" * 60)
        
        results = {}
        
        # Run all evaluations
        try:
            results["ground_truth_accuracy"] = self.evaluate_ground_truth_accuracy()
        except Exception as e:
            print(f"Ground truth accuracy evaluation failed: {e}")
        
        try:
            results["response_quality"] = self.evaluate_response_quality()
        except Exception as e:
            print(f"Response quality evaluation failed: {e}")
        
        try:
            results["price_prediction_handling"] = self.evaluate_price_prediction_handling()
        except Exception as e:
            print(f"Price prediction evaluation failed: {e}")
        
        try:
            results["response_time"] = self.evaluate_response_time()
        except Exception as e:
            print(f"Response time evaluation failed: {e}")
        
        self.evaluation_results = results
        
        print("\nComprehensive evaluation completed!")
        return results
    
    
    # Generate evaluation report 
    def generate_report(self, results: Dict[str, EvaluationResult], output_file: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        """
        if output_file is None:
            output_file = f"rag_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report = self._create_markdown_report(results)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Evaluation report saved to: {output_file}")
        return output_file
    
    def _create_markdown_report(self, results: Dict[str, EvaluationResult]) -> str:
        """Create a markdown formatted evaluation report according to user specifications."""
        
        # Header
        report = f"""# RAG Chatbot Evaluation Report

        **Generated Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        **Data File:** {self.data_file}

        ## Executive Summary

        """
        
        # Overall scores in specified order
        if "ground_truth_accuracy" in results:
            report += f"- **Ground Truth Accuracy:** {results['ground_truth_accuracy'].score:.3f}\n"
        if "response_quality" in results:
            report += f"- **Response Quality:** {results['response_quality'].score:.3f}\n"
        if "price_prediction_handling" in results:
            report += f"- **Price Prediction Handling:** {results['price_prediction_handling'].score:.3f}\n"
        if "response_time" in results:
            report += f"- **Response Time:** {results['response_time'].score:.3f}\n"
        
        # Overall RAG performance
        if results:
            avg_overall = statistics.mean([r.score for r in results.values()])
            report += f"\n**Overall RAG Performance:** {avg_overall:.3f}\n"
        
        # Test cases used for evaluation
        report += f"""
        ## Test Cases Used for Evaluation

        Total test cases: **{len(self.test_cases)}**

        """
        
        # List test cases by category
        categories = {}
        for tc in self.test_cases:
            if tc.category not in categories:
                categories[tc.category] = []
            categories[tc.category].append(tc.question)
        
        for category, questions in categories.items():
            report += f"**{category.replace('_', ' ').title()}** ({len(questions)} cases):\n"
            for q in questions:
                report += f"- {q}\n"
            report += "\n"
        
        # Detailed Results
        report += "## Detailed Results\n\n"
        
        # Ground Truth Accuracy
        if "ground_truth_accuracy" in results:
            report += self._format_ground_truth_section(results["ground_truth_accuracy"])
        
        # Response Quality  
        if "response_quality" in results:
            report += self._format_response_quality_section(results["response_quality"])
        
        # Price Prediction Handling
        if "price_prediction_handling" in results:
            report += self._format_price_prediction_section(results["price_prediction_handling"])
        
        # Response Time
        if "response_time" in results:
            report += self._format_response_time_section(results["response_time"])
        
        return report
    
    def _format_ground_truth_section(self, result: EvaluationResult) -> str:
        """Format ground truth accuracy section."""
        section = f"""### Ground Truth Accuracy

        **How it's derived:** Compares generated responses to manually created ground truth answers using semantic similarity (60%) combined with keyword matching (25%) and housing area mention accuracy (15%).

        **Score:** {result.score:.3f}

        """
        return section
    
    def _format_response_quality_section(self, result: EvaluationResult) -> str:
        """Format response quality section."""
        section = f"""### Response Quality

        **How it's derived:** Evaluates keyword presence (50%), area mention accuracy (30%), and response length appropriateness (20%).

        **Score:** {result.score:.3f}

        **Sample Generated Answers:**

        """
        # Add 3 random samples from detailed results
        if "detailed_results" in result.details and result.details["detailed_results"]:
            import random
            samples = random.sample(result.details["detailed_results"], min(2, len(result.details["detailed_results"])))
            
            for i, sample in enumerate(samples, 1):
                section += f"**Sample {i}:**\n"
                section += f"- **Question:** {sample['question']}\n"
                section += f"- **Response:** {sample['response']}\n"
                section += f"- **Quality Score:** {sample.get('overall_score', 'N/A'):.3f}\n\n"
        
        return section
    
    def _format_price_prediction_section(self, result: EvaluationResult) -> str:
        """Format price prediction handling section."""
        section = f"""### Price Prediction Handling

        **How it's derived:** Measures proper redirection to Analytics Dashboard (70%) and avoidance of making actual predictions (30%).

        **Score:** {result.score:.3f}

        **Sample Generated Answers for Price Prediction:**

        """
        # Add samples from price prediction detailed results
        if "detailed_results" in result.details and result.details["detailed_results"]:
            for i, sample in enumerate(result.details["detailed_results"], 1):
                section += f"**Sample {i}:**\n"
                section += f"- **Question:** {sample['question']}\n"
                section += f"- **Response:** {sample['response']}\n"
                section += f"- **Redirect Score:** {sample.get('redirect_score', 'N/A'):.3f}\n\n"
        
        return section
    
    def _format_response_time_section(self, result: EvaluationResult) -> str:
        """Format response time section."""
        section = f"""### Response Time

        **How it's derived:** Measures average response time in seconds. Score: ≤2s = 1.0, ≤5s = 0.8, ≤10s = 0.6, >10s = 0.4.

        **Score:** {result.score:.3f}

        """
        if "average_time" in result.details:
            section += f"**Average Response Time:** {result.details['average_time']:.2f} seconds\n\n"
        
        return section
    
    def print_summary(self, results: Dict[str, EvaluationResult]):
        """Print a quick summary of evaluation results."""
        print("\n" + "=" * 60)
        print("RAG EVALUATION SUMMARY")
        print("=" * 60)
        
        for metric_name, result in results.items():
            print(f"{metric_name.replace('_', ' ').title()}: {result.score:.3f}")
        
        if results:
            overall_avg = statistics.mean([r.score for r in results.values()])
            print(f"\nOverall Performance: {overall_avg:.3f}")
        
        print("=" * 60)


def main():
    """Main function for running RAG evaluation."""
    
    # Initialize evaluator
    data_file = "../data/hdb_rag/singapore_hdb_data.json"
    evaluator = RAGEvaluator(data_file)
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_full_pipeline()
    
    # Generate report
    report_file = evaluator.generate_report(results)
    
    # Print summary
    evaluator.print_summary(results)
    
    print(f"\nDetailed report available at: {report_file}")


if __name__ == "__main__":
    main()