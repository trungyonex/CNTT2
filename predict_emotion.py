import torch.nn as nn
import torch
from torchvision import transforms
import config
import cv2
import numpy as np
import os

class Emotic(nn.Module):
#   ''' Emotic Model'''
    def __init__(self, num_context_features, num_body_features):
        super(Emotic,self).__init__()
        self.num_context_features = num_context_features
        self.num_body_features = num_body_features
        self.fc1 = nn.Linear((self.num_context_features + num_body_features), 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_cat = nn.Linear(256, 26)
        self.fc_cont = nn.Linear(256, 3)
        self.relu = nn.ReLU()


    def forward(self, x_context, x_body):
        context_features = x_context.view(-1, self.num_context_features)
        body_features = x_body.view(-1, self.num_body_features)
        fuse_features = torch.cat((context_features, body_features), 1)
        fuse_out = self.fc1(fuse_features)
        fuse_out = self.bn1(fuse_out)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)    
        cat_out = self.fc_cat(fuse_out)
        cont_out = self.fc_cont(fuse_out)
        return cat_out, cont_out

class Emotic_PreData():
#   ''' Custom Emotic dataset class. Use preprocessed data stored in npy files. '''
    def __init__(self, x_context, x_body, transform, context_norm, body_norm):
        super(Emotic_PreData,self).__init__()
        self.x_context = x_context
        self.x_body = x_body
        self.transform = transform 
        self.context_norm = transforms.Normalize(context_norm[0], context_norm[1])  # Normalizing the context image with context mean and context std
        self.body_norm = transforms.Normalize(body_norm[0], body_norm[1])           # Normalizing the body image with body mean and body std

    def __len__(self):
        return len(self.y_cat)
  
    def __oneimage__(self):
        image_context = self.x_context
        image_body = self.x_body
        return self.context_norm(self.transform(image_context)).unsqueeze(0), self.body_norm(self.transform(image_body)).unsqueeze(0)
    def __getimage__(self, index):
        image_context = self.x_context[index]
        image_body = self.x_body[index]
        return self.context_norm(self.transform(image_context)).unsqueeze(0), self.body_norm(self.transform(image_body)).unsqueeze(0)

def predict_emotion(bbox, filename):
    if len(bbox) == 1:
        emotions, vad = predict_one_person(bbox, filename)
        image_context = cv2.imread(filename)
        image_context = cv2.rectangle(image_context, (bbox[0][0], bbox[0][1]),(bbox[0][2] , bbox[0][3]), (255, 0, 0), 3)
        cv2.putText(image_context, vad, (bbox[0][0], bbox[0][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        for i, emotion in enumerate(emotions):
            cv2.putText(image_context, emotion, (bbox[0][0], bbox[0][1] + (i+1)*12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        name_result = 'result_'+filename.replace('/', '-')
        cv2.imwrite(os.path.join(config.PREDICTED_PATH, name_result), image_context)
        return name_result, emotions, vad
    elif len(bbox) > 1:
        list_emotions, list_vad = predict_many_people(bbox, filename)
        image = cv2.imread(filename)
        for i in range(len(bbox)):
            person = 'person_'+str(i)
            cv2.rectangle(image, (bbox[i][0], bbox[i][1]),(bbox[i][2] , bbox[i][3]), (255, 0, 0), 3)
            cv2.putText(image, person, (bbox[i][0], bbox[i][1] - 17), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(image, list_vad[i], (bbox[i][0], bbox[i][1] - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            for j, emotion in enumerate(list_emotions[i]):
                cv2.putText(image, emotion, (bbox[i][0], bbox[i][1] + (j+1)*12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        name_result = 'result_'+filename.replace('/', '-')
        cv2.imwrite(os.path.join(config.PREDICTED_PATH, name_result), image)
        return name_result, list_emotions, list_vad
    else:
        return "Image don't have any people"

def predict_one_person(bbox, filename):
    img = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
    body = img[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]].copy() 

    context = cv2.resize(img, (224,224))
    body_cv = cv2.resize(body, (128,128))

    context_arr = np.array(context)
    body_arr = np.array(body_cv)

    model_context = torch.load(os.path.join(config.MODEL_PATH, config.MODEL_CONTEXT))
    model_body = torch.load(os.path.join(config.MODEL_PATH, config.MODEL_BODY))
    emotic_model = torch.load(os.path.join(config.MODEL_PATH, config.MODEL_EMOTIC))

    test_transform = transforms.Compose([transforms.ToPILImage(), 
                                     transforms.ToTensor()])
    test_dataset = Emotic_PreData(context_arr, body_arr, test_transform, config.CONTEXT_NORM, config.BODY_NORM)
    images_context, images_body = test_dataset.__oneimage__()

    device = config.DEVICE

    model_context.to(device)
    model_body.to(device)
    emotic_model.to(device)

    model_context.eval()
    model_body.eval()
    emotic_model.eval()

    images_context = images_context.to(device)
    images_body = images_body.to(device)
            
    pred_context = model_context(images_context)
    pred_body = model_body(images_body)
    pred_cat, pred_cont = emotic_model(pred_context, pred_body)
    return standard_result(pred_cat, pred_cont)

def standard_result(pred_cat, pred_cont):
    pred_cat = pred_cat.squeeze(0)
    pred_cont = pred_cont.squeeze(0).to("cpu").data.numpy()

    thresholds = torch.FloatTensor(np.load(os.path.join(config.MODEL_PATH, config.THRESHOLDS_PATH))).to(config.DEVICE)
    bool_cat_pred = torch.gt(pred_cat, thresholds)

    cat_emotions = list()
    for i in range(len(bool_cat_pred)):
        if bool_cat_pred[i] == True:
            cat_emotions.append(config.IND2CAT[i])

    pred_cont = 10*pred_cont
    write_text_vad = list()
    for continuous in pred_cont:
        write_text_vad.append(str('%.1f' %(continuous)))
    write_text_vad = 'vad ' + ' '.join(write_text_vad)
    return cat_emotions, write_text_vad

def predict_many_people(bbox, filename):
    img = cv2.cvtColor(cv2.imread(filename),cv2.COLOR_BGR2RGB)
    context_arr = []
    body_arr = []
    for i in bbox:
        context = cv2.resize(img, (224,224))
        context_arr.append(np.array(context))
        
        body = img[i[1]:i[3],i[0]:i[2]].copy()
        body_cv = cv2.resize(body, (128,128))
        
        body_arr.append(np.array(body_cv))
    
    model_context = torch.load(os.path.join(config.MODEL_PATH, config.MODEL_CONTEXT))
    model_body = torch.load(os.path.join(config.MODEL_PATH, config.MODEL_BODY))
    emotic_model = torch.load(os.path.join(config.MODEL_PATH, config.MODEL_EMOTIC))

    test_transform = transforms.Compose([transforms.ToPILImage(), 
                                     transforms.ToTensor()])
    test_dataset = Emotic_PreData(context_arr, body_arr, test_transform, config.CONTEXT_NORM, config.BODY_NORM)

    device = config.DEVICE

    model_context.to(device)
    model_body.to(device)
    emotic_model.to(device)

    model_context.eval()
    model_body.eval()

    list_cat_emotions = []
    list_vad = []
    for i in range(len(bbox)):
        images_context, images_body = test_dataset.__getimage__(i)
        with torch.no_grad():
            images_context = images_context.to(device)
            images_body = images_body.to(device)

            pred_context = model_context(images_context)
            pred_body = model_body(images_body)
            pred_cat, pred_cont = emotic_model(pred_context, pred_body)

            cat_emotions, write_text_vad = standard_result(pred_cat, pred_cont)
        list_cat_emotions.append(cat_emotions)
        list_vad.append(write_text_vad)
    return list_cat_emotions, list_vad

def predict_video(bbox, img, model_context, model_body, emotic_model):
    if len(bbox) == 1:
        body = img[bbox[0][1]:bbox[0][3],bbox[0][0]:bbox[0][2]].copy() 

        context = cv2.resize(img, (224,224))
        body_cv = cv2.resize(body, (128,128))

        context_arr = np.array(context)
        body_arr = np.array(body_cv)

        test_transform = transforms.Compose([transforms.ToPILImage(), 
                                        transforms.ToTensor()])
        test_dataset = Emotic_PreData(context_arr, body_arr, test_transform, config.CONTEXT_NORM, config.BODY_NORM)
        images_context, images_body = test_dataset.__oneimage__()

        device = config.DEVICE

        images_context = images_context.to(device)
        images_body = images_body.to(device)
                
        pred_context = model_context(images_context)
        pred_body = model_body(images_body)
        pred_cat, pred_cont = emotic_model(pred_context, pred_body)
        return standard_result(pred_cat, pred_cont)
    elif len(bbox) > 1:
        context_arr = []
        body_arr = []
        for i in bbox:
            context = cv2.resize(img, (224,224))
            context_arr.append(np.array(context))
            
            body = img[i[1]:i[3],i[0]:i[2]].copy()
            body_cv = cv2.resize(body, (128,128))
            
            body_arr.append(np.array(body_cv))

        test_transform = transforms.Compose([transforms.ToPILImage(), 
                                        transforms.ToTensor()])
        test_dataset = Emotic_PreData(context_arr, body_arr, test_transform, config.CONTEXT_NORM, config.BODY_NORM)

        device = config.DEVICE

        list_cat_emotions = []
        list_vad = []
        for i in range(len(bbox)):
            images_context, images_body = test_dataset.__getimage__(i)
            with torch.no_grad():
                images_context = images_context.to(device)
                images_body = images_body.to(device)

                pred_context = model_context(images_context)
                pred_body = model_body(images_body)
                pred_cat, pred_cont = emotic_model(pred_context, pred_body)

                cat_emotions, write_text_vad = standard_result(pred_cat, pred_cont)
            list_cat_emotions.append(cat_emotions)
            list_vad.append(write_text_vad)
        return list_cat_emotions, list_vad    
    else:
        return 0
