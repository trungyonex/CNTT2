import torch

cat = ['Affection', 'Anger', 'Annoyance', 'Anticipation', 'Aversion', 'Confidence', 'Disapproval', 'Disconnection', \
          'Disquietment', 'Doubt/Confusion', 'Embarrassment', 'Engagement', 'Esteem', 'Excitement', 'Fatigue', 'Fear','Happiness', \
          'Pain', 'Peace', 'Pleasure', 'Sadness', 'Sensitivity', 'Suffering', 'Surprise', 'Sympathy', 'Yearning']
ind2cat = {}
for idx, emotion in enumerate(cat):
    ind2cat[idx] = emotion

IND2CAT = ind2cat

vad = ['Valence', 'Arousal', 'Dominance']
ind2vad = {}
for idx, continuous in enumerate(vad):
    ind2vad[idx] = continuous

IND2VAD = ind2vad
MODEL_PATH = 'models'
MODEL_CONTEXT = 'model_context1.pth'
MODEL_BODY = 'model_body1.pth'
MODEL_EMOTIC = 'model_emotic1.pth'
THRESHOLDS_PATH = 'thresholds.npy'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

context_mean = [0.4690646, 0.4407227, 0.40508908]
context_std = [0.2514227, 0.24312855, 0.24266963]
body_mean = [0.43832874, 0.3964344, 0.3706214]
body_std = [0.24784276, 0.23621225, 0.2323653]
CONTEXT_NORM = [context_mean, context_std]
BODY_NORM = [body_mean, body_std]
PREDICTED_PATH = 'static/predicted'