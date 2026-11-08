LABELS = ["supervised", "unsupervised", "reinforcement"]
LETTERS = {"A": "supervised", "B": "unsupervised", "C": "reinforcement", "D": "not_ml"}

DOMAINS = ["healthcare", "finance", "retail", "robotics", "gaming", "manufacturing", "agriculture", "cybersecurity"]

KNOWN_ALGOS = {
    "supervised": ["logistic regression", "linear regression", "random forest", "gradient boost", "xgboost", "lightgbm",
                   "svm", "support vector", "decision tree", "naive bayes", "k-nearest", "knn", "neural network", "cnn",
                   "convolutional", "bert", "transformer", "lstm", "rnn", "resnet", "catboost", "ridge", "lasso",
                   "mlp", "classifier", "regression", "u-net", "yolo", "efficientnet", "gbm"],
    "unsupervised": ["k-means", "kmeans", "dbscan", "hierarchical", "agglomerative", "pca", "principal component",
                     "autoencoder", "gaussian mixture", "gmm", "isolation forest", "t-sne", "umap", "apriori",
                     "association rule", "lda", "latent dirichlet", "spectral", "one-class", "fp-growth", "som",
                     "self-organizing", "hdbscan", "clustering", "topic model", "mean shift", "vae", "variational"],
    "reinforcement": ["q-learning", "dqn", "deep q", "ppo", "proximal policy", "a2c", "a3c", "actor-critic",
                      "sac", "soft actor", "ddpg", "td3", "sarsa", "policy gradient", "reinforce", "monte carlo tree",
                      "mcts", "multi-armed bandit", "bandit", "thompson sampling", "ucb", "trpo", "td learning",
                      "alphazero", "muzero", "temporal difference"],
}

FEW_SHOT = [
    ("A bank has 50,000 past loan applications each marked as defaulted or repaid and wants to score new applicants.", "A"),
    ("An online shop wants to discover natural groups of customers from purchase histories that have no tags.", "B"),
    ("A warehouse robot must learn to navigate aisles by trying moves and getting points for fast, collision-free deliveries.", "C"),
    ("Radiologists annotated 8,000 chest X-rays as pneumonia or normal; the hospital wants to flag new scans.", "A"),
    ("A security team wants to spot unusual login patterns in server logs where no attacks have been marked.", "B"),
    ("A thermostat should learn a heating schedule by adjusting temperature and receiving feedback on energy cost and comfort.", "C"),
]

# 20 hand-labelled gold scenarios (author-labelled), including deliberately tricky ones.
BENCHMARK = [
    {"id": "B01", "domain": "healthcare", "label": "supervised", "text": "A clinic has 12,000 patient records with blood tests and a confirmed diabetes diagnosis for each, and wants to predict diabetes for new patients."},
    {"id": "B02", "domain": "healthcare", "label": "unsupervised", "text": "Researchers want to find previously unknown subtypes of a disease by grouping patients' gene-expression profiles."},
    {"id": "B03", "domain": "healthcare", "label": "reinforcement", "text": "A system must choose insulin doses step by step for a simulated patient, being rewarded for keeping glucose in a safe range over the day."},
    {"id": "B04", "domain": "finance", "label": "supervised", "text": "Predict next month's credit card spend for each customer from two years of their historical monthly spend figures."},
    {"id": "B05", "domain": "finance", "label": "unsupervised", "text": "A fintech wants to flag strange transactions, but none of its millions of past transactions are marked as fraud or legitimate."},
    {"id": "B06", "domain": "finance", "label": "reinforcement", "text": "A trading agent buys and sells a stock each minute in a market simulator and is scored by its final portfolio value."},
    {"id": "B07", "domain": "retail", "label": "supervised", "text": "Given product reviews already tagged positive, neutral or negative by staff, automatically tag incoming reviews."},
    {"id": "B08", "domain": "retail", "label": "unsupervised", "text": "A supermarket wants to know which products tend to be bought together from millions of receipts."},
    {"id": "B09", "domain": "retail", "label": "reinforcement", "text": "A website picks which of five banner designs to show each visitor and must learn over time which one earns the most clicks."},
    {"id": "B10", "domain": "robotics", "label": "reinforcement", "text": "A four-legged robot must learn to walk across rough terrain by trial and error, earning credit for forward progress without falling."},
    {"id": "B11", "domain": "robotics", "label": "supervised", "text": "A robot arm should copy grasps from 3,000 recorded human demonstrations, each pairing a camera image with the correct gripper pose."},
    {"id": "B12", "domain": "gaming", "label": "reinforcement", "text": "Build an AI that becomes strong at a board game purely by playing millions of games against itself."},
    {"id": "B13", "domain": "gaming", "label": "unsupervised", "text": "A game studio wants to discover distinct player styles from gameplay telemetry without any predefined categories."},
    {"id": "B14", "domain": "manufacturing", "label": "supervised", "text": "Inspectors labelled 20,000 photos of circuit boards as defective or fine; the factory wants an automatic visual inspector."},
    {"id": "B15", "domain": "manufacturing", "label": "unsupervised", "text": "Compress 400 correlated sensor readings from a production line into a handful of underlying factors for monitoring."},
    {"id": "B16", "domain": "agriculture", "label": "supervised", "text": "Estimate wheat yield per field from satellite imagery, using past seasons where the harvested yield was measured."},
    {"id": "B17", "domain": "agriculture", "label": "reinforcement", "text": "An irrigation controller decides daily how much water to release and learns from crop growth and water cost feedback over seasons."},
    {"id": "B18", "domain": "cybersecurity", "label": "supervised", "text": "Classify emails as phishing or safe using a corpus where security analysts have marked each email."},
    {"id": "B19", "domain": "cybersecurity", "label": "unsupervised", "text": "Group thousands of malware samples into families based on their behaviour, without knowing the families in advance."},
    {"id": "B20", "domain": "cybersecurity", "label": "reinforcement", "text": "An automated defender chooses which network ports to close during a simulated attack and is scored on how many assets stay safe."},
]

# 10 deliberately tricky / mixed-paradigm scenarios: domain cue words point to the wrong paradigm.
HARD_BENCHMARK = [
    {"id": "H01", "domain": "retail", "label": "supervised", "text": "A marketing team first clusters customers into personas, then has staff name each persona, and the actual deliverable is a model that assigns each new sign-up to one of those named personas."},
    {"id": "H02", "domain": "retail", "label": "supervised", "text": "A research team runs k-means on 50,000 support tickets, has agents name the 12 resulting clusters, and then ships a model that routes each incoming ticket to one of those 12 named queues."},
    {"id": "H03", "domain": "retail", "label": "reinforcement", "text": "A chatbot is fine-tuned by having people rate its replies and training it to produce replies that earn higher ratings."},
    {"id": "H04", "domain": "finance", "label": "reinforcement", "text": "An ad marketplace sets its price every hour; each price changes how many advertisers bid in the following hours, and the goal is the highest revenue over the year."},
    {"id": "H05", "domain": "manufacturing", "label": "supervised", "text": "A factory has 200 images marked defective or normal plus 50,000 unmarked ones; it pre-trains on the unmarked images and then predicts defective vs normal for each new part."},
    {"id": "H06", "domain": "manufacturing", "label": "unsupervised", "text": "An energy company wants to predict which of its smart meters are behaving abnormally, but it has never recorded a single confirmed meter fault."},
    {"id": "H07", "domain": "healthcare", "label": "unsupervised", "text": "Before any diagnosis model is built, a hospital wants to compress 3,000 lab measurements per patient into a handful of summary scores."},
    {"id": "H08", "domain": "gaming", "label": "supervised", "text": "A game studio records which move a human expert chose in each of 50,000 game states and trains an agent to pick the same move in new states; the agent never receives a score."},
    {"id": "H09", "domain": "robotics", "label": "supervised", "text": "A self-driving startup has 500 hours of expert driving with the steering angle recorded for every camera frame and wants the car to copy that driving."},
    {"id": "H10", "domain": "finance", "label": "supervised", "text": "Last year a bank's analysts placed every customer into one of five segments by hand; now new customers should be placed into those same five segments automatically."},
]

ALL_BENCHMARK = BENCHMARK + HARD_BENCHMARK

MODEL_CARD = [
    ["Model", "DistilBERT-base-uncased fine-tuned (Keras 3, torch backend) + 3-way softmax head"],
    ["Intended use", "Teaching aid: suggest whether a described ML problem is supervised, unsupervised or reinforcement learning"],
    ["Out of scope", "Production ML system design decisions, non-English text, hybrid paradigms (semi/self-supervised, imitation)"],
    ["Training data", "200 LLM-generated scenarios across 8 domains, 80/20 stratified split, seed 42"],
    ["Evaluation", "Held-out synthetic split + 20 hand-labelled gold scenarios + 10 tricky mixed-paradigm scenarios; per-domain accuracy"],
    ["Known failure modes", "Bandit / A-B testing problems and association-rule problems are often called supervised; relies on cue words such as 'predict' or 'flag'"],
    ["Safety", "Input guardrail (prompt injection, harmful content, length), PII redaction, off-topic 'D' class from the LLM"],
]

LICENCES = [
    ["distilbert-base-uncased", "Apache-2.0", "Hugging Face / original BERT by Google"],
    ["gpt-4.1-mini (OpenAI API)", "Proprietary - API terms", "Outputs owned by the user per OpenAI terms; no model weights shipped"],
    ["gemini-3.1-flash-image (Nano Banana)", "Proprietary - API terms", "Generated images carry SynthID watermark; for illustration only"],
    ["Keras 3, PyTorch, transformers, scikit-learn", "Apache-2.0 / BSD", "Open-source libraries"],
    ["Synthetic dataset (200)", "CC-BY-4.0 (this project)", "LLM-generated; no personal data"],
    ["30 hand-labelled scenarios", "CC-BY-4.0 (this project)", "Written by the student author"],
]

PRESETS = [
    "A hospital has thousands of labelled MRI scans marked tumour or no tumour and wants to screen new scans automatically.",
    "An e-commerce company wants to segment its customers into groups based on browsing behaviour, with no predefined segments.",
    "A delivery drone must learn the most energy-efficient route through changing winds by trying actions and receiving rewards.",
    "A farmer wants to know which of 10 fertiliser mixes works best, trying them on plots each week and observing yield.",
]

SYNTHETIC_LIMITS = [
    "All 200 training scenarios are written by one LLM (gpt-4.1-mini), so DistilBERT learns that model's phrasing and stereotypes, not real-world text.",
    "Synthetic scenarios tend to contain obvious cue words (labelled, reward, groups) that make the task look easier than it is.",
    "Label balance is forced by the prompt (≈1/3 per paradigm); real problems are dominated by supervised learning.",
    "Ambiguous and hybrid cases (semi-supervised, self-supervised, bandits, imitation learning) are under-represented.",
    "The generator labels its own scenarios, so any LLM labelling mistakes become 'ground truth' for DistilBERT.",
    "Only English and 8 hand-picked domains are covered; the per-domain test sets are small, so accuracy gaps are noisy.",
]
