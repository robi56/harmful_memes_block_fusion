import argparse,pprint

from multiprocessing import set_forkserver_preload
import os
from string import digits
import pandas as pd
import torch
import json
from PIL import Image
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertTokenizer
from torch import nn
from transformers import BertModel
import numpy as np
from block import fusions
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score,accuracy_score, confusion_matrix
import timm 
import torchvision.transforms as transforms


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=2)
    parser.add_argument("--bert_model", type=str, default="bert-base-cased", choices=["bert-base-uncased", "bert-large-uncased","bert-base-cased"])
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--dev_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--nepochs", type=int, default=50)
    parser.add_argument("--checkpoint_path", type=str, default="models/checkpoint.pt")
    parser.add_argument("--result_path", type=str, default="results/result.jsonl")
    parser.add_argument("--nsamples", type=int, default=-1)



labels = {'hero': 0,
          'villain': 1,
          'victim': 2,
          'other': 3
          }
image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def bert_tokenizer(text,tokenizer,max_length=512):
    try:
        tokenized = tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        return tokenized
    except Exception as e:
        print(e)
        print(text)
        tokenized = tokenizer("UNK", padding='max_length', max_length=512, truncation=True, return_tensors="pt")
        return tokenized


class Attention(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights

class Dataset(torch.utils.data.Dataset):

    def __init__(self, df,tokenizer):
        self.labels = [label for label in df['label']]
        processed_texts = []
        processed_entities= []
        processed_images =[]
        for i, j in df.iterrows():
            text = str(j['text'])
            text = text.replace("\n", " ")
            processed_texts.append(text)
            processed_entities.append(str(j['entity']))
            processed_images.append(str(j['image']))

        self.texts = [bert_tokenizer(text,tokenizer, max_length=256) for text in processed_texts]
        self.entities = [bert_tokenizer(text,tokenizer) for text in processed_entities]
        self.classes = set(np.asarray(self.labels))
        self.images = [image_transforms(Image.open(img_path).convert('RGB')) for img_path in processed_images]

    def classes(self):
        return self.labels()

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
    def get_batch_entities(self, idx):
        # Fetch a batch of inputs
        return self.entities[idx]
    def get_batch_images(self, idx):
        # Fetch a batch of inputs
        return self.images[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_entities = self.get_batch_entities(idx)
        batch_images = self.get_batch_images(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts,batch_entities, batch_images, batch_y



class BertClassifier_Update(nn.Module):

    def __init__(self,bert_model, dropout=0.5):

        super(BertClassifier_Update, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model)
        self.entity = BertModel.from_pretrained(bert_model)
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Linear(self.vit.head.in_features, 512)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 4)
        self.relu = nn.ReLU()
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(self.bert.config.hidden_size,512)
        self.fc2 = nn.Linear(self.entity.config.hidden_size,512)
        self.fc3 = nn.Linear(512,4)
        self.mm =fusions.Block([512,512], 512)
        self.attention=Attention(512)

    def forward(self, input_id, mask, input_id_entity, mask_entity, img):
        #print(input_id.shape)
        #print(mask.shape)
        
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        _, pooled_output_entity = self.bert(input_ids= input_id_entity, attention_mask=mask_entity,return_dict=False)
        text_out = self.fc1(pooled_output)
        entity_out = self.fc2(pooled_output_entity)
        img_out = self.vit(img)
        attn_image= entity_out.reshape(-1,1,img_out.shape[1])
        attn_text= entity_out.reshape(-1,1,text_out.shape[1])

        output, weights=self.attention(attn_image,attn_text)
        output = output.reshape(-1,output.shape[2])
        # merged=self.bn1(merged)
        # entity_out=self.bn2(entity_out)
        inputs = [entity_out,output]
        fused=self.mm(inputs)

        dropout_output = self.dropout(fused)
        linear_output = self.fc3(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


def eval(model, dataloader, criterion, device, save_prediction=False):
    total_acc_val = 0
    total_loss_val = 0
    predictions=[]
    labels=[]
    result ={}
    with torch.no_grad():

        for val_input,val_entity, image, val_label in dataloader:

            val_label = val_label.to(device)
            mask = val_input['attention_mask'].to(device)
            input_id = val_input['input_ids'].squeeze(1).to(device)
            mask_entity = val_entity['attention_mask'].to(device)
            input_id_entity = val_entity['input_ids'].squeeze(1).to(device)
            image = image.to(device)



            output = model(input_id, mask,input_id_entity,mask_entity, image)
            if save_prediction:
                if len(predictions)>0:
                    predictions=torch.concat([predictions,output.argmax(dim=1)] )
                else:
                    predictions=output.argmax(dim=1)
                prediction_list= predictions.cpu().numpy()

                d={}
                d["test"]={"pred":prediction_list.tolist()}

                with open("test_result.json", "w") as f:
                    json.dump(d,f)
                
                return 0.0, 0.0

            try:
                batch_loss = criterion(output, val_label)
                total_loss_val += batch_loss.item()
            except Exception as e:
                print(e)
            acc = (output.argmax(dim=1) == val_label).sum().item()
            total_acc_val += acc
            if len(predictions)>0:
                    predictions=torch.concat([predictions,output.argmax(dim=1)] )
                    labels = torch.concat([labels, val_label])
            else:
                    predictions=output.argmax(dim=1)
                    labels = val_label

        prediction_list= predictions.cpu().numpy()
        label_list= labels.cpu().numpy()
        print(classification_report(label_list,prediction_list, digits=3))
        result["macro-f1"]= f1_score(label_list, prediction_list, average="macro")
        result["precision"]= precision_score(label_list, prediction_list, average="macro")
        result["recall-score"]= recall_score(label_list, prediction_list, average="macro")
        result["accuracy"]= accuracy_score(label_list, prediction_list)
        result["report"]=classification_report(label_list, prediction_list, digits=3,output_dict=False)
        result["cm"]=confusion_matrix(label_list, prediction_list)
        result["total_loss"]=total_loss_val

        return result




def train(args):
    
    if os.path.isdir("models"):
        pass 
    else:
       os.makedirs("models", exist_ok=True)

    
    if os.path.isdir("results"):
        pass 
    else:
       os.makedirs("results", exist_ok=True)
    
    train_data = pd.read_csv(args.train_file, sep="\t")
    val_data = pd.read_csv(args.dev_file, sep="\t")
    test_data = pd.read_csv(args.test_file, sep="\t")

    if args.nsamples>0:
        train_data=train_data[0:args.nsamples]
        val_data=val_data[0:args.nsamples]
        test_data=test_data[0:args.nsamples]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    

    train, val, test = Dataset(train_data,tokenizer), Dataset(val_data,tokenizer), Dataset(test_data,tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_sz, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=args.batch_sz)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=args.batch_sz, drop_last=True)


    epochs =  args.nepochs
    checkpoint_path=args.checkpoint_path
    LR = 1e-6

    model = BertClassifier_Update(args.bert_model)
    optimizer = Adam(model.parameters(), lr=LR)
    if os.path.exists(checkpoint_path):
        print("model load from checkpoint: "+ checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.train()
        
    
    use_cuda = torch.cuda.is_available()
    #use_cuda=False
    device = torch.device("cuda" if use_cuda else "cpu")
    #device='cpu'

    criterion = nn.CrossEntropyLoss()
    

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(0, epochs):
        total_acc_train = 0
        total_loss_train = 0
        predictions=[]
        labels=[]
        performance={}
        performance["epoch"]=str(epoch_num)

        for train_input, train_entity, image, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            mask_entity = train_entity['attention_mask'].to(device)
            input_id_entity = train_entity['input_ids'].squeeze(1).to(device)
            image=image.to(device)

            output = model(input_id, mask,input_id_entity, mask_entity, image)
            try:
                batch_loss = criterion(output, train_label)
                total_loss_train += batch_loss.item()
            except Exception as e:
                print(e)
            
            if len(predictions)>0:
                    predictions=torch.concat([predictions,output.argmax(dim=1)] )
                    labels = torch.concat([labels, train_label])
            else:
                    predictions=output.argmax(dim=1)
                    labels = train_label          


            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        
        print("Training report after epoch "+ str(epoch_num))
        prediction_list= predictions.cpu().numpy()
        label_list= labels.cpu().numpy()
        print(classification_report(label_list,prediction_list, digits=3))
        result_train={}
        result_train["macro-f1"]= f1_score(label_list, prediction_list, average="macro")
        result_train["precision"]= precision_score(label_list, prediction_list, average="macro")
        result_train["recall-score"]= recall_score(label_list, prediction_list, average="macro")
        result_train["accuracy"]= accuracy_score(label_list, prediction_list)
        result_train["report"]=classification_report(label_list, prediction_list, digits=3,output_dict=False)
        result_train["cm"]=confusion_matrix(label_list, prediction_list)
        result_train["total_loss"]=total_loss_train
        
        performance["train"]=result_train
        
        total_acc_val = 0
        total_loss_val = 0

        print("Validation report  after epoch "+str(epoch_num))
        result_val =eval(model, val_dataloader, criterion, device)
        performance["val"]=result_val
       
        print("Test report  after epoch "+str(epoch_num))

        result_test =eval(model, test_dataloader, criterion, device)

        performance["test"]=result_test
        
        pretty_print_json = pprint.pformat(performance).replace("'", '"')
        with open(args.result_path,'a+') as f:
            f.write(pretty_print_json)
            f.write("\n\n")
         

      
        if int(epoch_num/2)>0 and int(epoch_num%2)==0:
            print("model save to ", checkpoint_path," after epoch ", str(epoch_num))
            torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)

    

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f} ')
    
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    

def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()