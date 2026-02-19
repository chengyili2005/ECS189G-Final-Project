
import re
from collections import Counter

class Evaluator:
    def __init__(self, prompt_template, question_key, answer_key, context_key):
        self.prompt_template = prompt_template
        self.question_key = question_key
        self.answer_key = answer_key
        self.context_key = context_key
        self.results = {}

    def normalize(self, text):
        text = text.lower() #lowercase everyting 
        text = re.sub(r'\b(a|an|the)\b', ' ', text) #remove less meaningful words, not sure if necessary 
        text = re.sub(r'[^\w\s]', '', text) #strip punctuation 
        text = ' '.join(text.split()) #standardize spacing 
        return text

    def exact_match(self, pred, gold):
        return int(self.normalize(pred) == self.normalize(gold))
    
    def token_f1(self, pred, gold):
        pred_tokens = self.normalize(pred).split()
        gold_tokens = self.normalize(gold).split()
        
        #potential edge cases
        if len(pred_tokens) == 0 and len(gold_tokens) == 0: 
            return 1.0
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(gold_tokens) #get num shared tokens
        num_shared = sum(common.values()) 
        
        #calculate precision and recall
        precision = num_shared / len(pred_tokens)
        recall = num_shared / len(gold_tokens)
        
        if precision + recall == 0: #edge case 
            return 0.0  
        
        return 2*precision*recall/(precision + recall) #calculate F1, and return 

    def generate(self, llm, prompt, max_new_tokens=64):
        output = llm(prompt, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
        return output[0]['generated_text'].strip()

    def evaluation(self, llm, dataset, model_name, preprocess_fn=None, max_samples=None):
        if max_samples: 
            dataset = dataset.select(range(min(max_samples, len(dataset)))) #get specified num samples to test on 
        
        em_scores = []
        f1_scores =[]
        predictions = []
        
        for example in dataset:
            if preprocess_fn: #function to flatten or otherwise preprocess text 
                example = preprocess_fn(example) 
            
            prompt = self.prompt_template.format( #fit to provided template, i.e Context: " "  ... Question: " " ... Answer: 
                question=example[self.question_key],
                context=example[self.context_key])
            
            pred = self.generate(llm, prompt) #get model output
            gold = example[self.answer_key] #get gold from dataset
            em_score = self.exact_match(pred, gold) #evaluate
            f1_score = self.token_f1(pred, gold)
            
            em_scores.append(em_score)
            f1_scores.append(f1_score)

            predictions.append({ #store results
                "question": example[self.question_key],
                "gold": gold,
                "pred": pred,
                "em_score": em_score,
                "f1_score": f1_score})
        
        # Compute average and store
        avg_em_score = sum(em_scores)/len(em_scores)
        avg_f1_score = sum(f1_score)/len(f1_score)
        
        self.results[model_name] = {
            "em_score": avg_em_score,
            "f1_score": avg_f1_score,
            "predictions": predictions,
            "n_samples": len(em_scores)
        }
        
        return avg_em_score, avg_f1_score

    def display_results(self):
      
        print("Results:")
        print("-"*50)

        print(f"{'Model':<20}{'EM Score':<15}{'F1 Score':<15}")
        print("-"*50)

        for model_name, data in self.results.items(): #print results for each model run with this evaluator 
            print(f"{model_name:<20}{data['em_score']:<15.4f}{data['f1_score']:<15.4f}")
        
        print("-"*50)
