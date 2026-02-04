# Narrative Consistency Reasoning over Long-Form Texts

This project addresses the problem of **global logical consistency checking in long narratives**.  
Given a complete novel (100k+ words) and a hypothetical backstory for a central character, the system determines whether the backstory is **logically consistent** with the narrative as a whole.

Built as part of the **Kharagpur Data Science Hackathon (Pathway Track A)**.

---

##  Problem Overview

Given a backstory and a story (evidence) from a novel, determine whether the backstory is logically consistent with the story.
The task focuses on logical compatibility, not semantic similarity.




---

##  Technologies Used

- Python  
- Pathway (data ingestion and orchestration)  
- Sentence-Transformers (vector embeddings)  
- Cosine similarity retrieval  
- LLM-based logical judge (NVIDIA NIM API)  
- Pandas / NumPy  


##  Approach Overview

Instead of fine-tuning model weights, the system relies on:

- Prompt-based logical reasoning  
- Claim-level backstory decomposition  
- Threshold-based aggregation over contradiction strength  

---

##  System Architecture

### Pipeline Overview

#### Data Ingestion
- Train/test CSVs containing backstories  
- Full novels loaded as raw `.txt` files (no truncation)

#### Long-Context Handling
- Novels split into overlapping semantic chunks  
- Sentence embeddings generated using `all-MiniLM-L6-v2`

#### Evidence Retrieval
- Cosine similarity search to retrieve top-K relevant passages  
- Retrieval constrained by book identity

#### Claim Decomposition
- Backstories split into independent logical claims  
- Each claim evaluated separately

#### Logical Consistency Judgement
- External LLM (LLaMA 3.1 via NVIDIA NIM API)  
- Strict contradiction-based prompting (no hallucination, no inference)

#### Aggregation
- If **any** claim contradicts the novel → label **0**  
- Otherwise → label **1**

---




A backstory is marked incorrect (0) only if it is logically impossible given the evidence.

📈 Results
Baseline prompt accuracy: ~62%
Threshold‑based policy accuracy: ~70%

---


##  Team Details
- Team Name:  
- Member 1 Charvi Sri Dadi
- Member 2  Monisha Reddi
- Member 3  Mokila Kshitij Reddy
- Member 4  Pavan Satwik

  ---
##  How to Run

```bash
pip install -r requirements.txt
python src/pipeline.py
