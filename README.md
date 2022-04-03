# harmful_memes_block_fusion
Pytorch code for CONSTRAINT-2022 :Shared Task - Task: Hero, Villain and Victim: Dissecting harmful memes for Semantic role labelling of entities 

## Task description 

 The objective is to classify for a given pair of a meme and an entity, whether the entity is being referenced as Hero vs. Villain vs. Victim vs. Other, within that meme. More information  https://constraint-lcs2.github.io/ 

## Features of the work 
- Used block fusion 
- Text Data Augmentation 
- Attention Mechanism 

## Dependencies 

block.bootstrap.pytorch==0.1.6 <br/>
transformers==4.10.1 <br/>
torchvision==0.12.0 <br/>
torch==1.11.0 <br/>
timm==0.5.4 <br/>
scikit-learn==1.0.2 <br/>
pandas==1.3.4 <br/>
fasttext==0.9.1 <br/>
sklearn-crfsuite==0.3.6

## Data Preparation 
Entity wise data preparation , the total converted data is equal to the total number of entities in the *.jsonl file. 

File: constraint22_dataset_covid19/annotations/train.jsonl

{"OCR": "Bernie or Elizabeth?\nBe informed.Compare them on the issues that matter.\nIssue: Who makes the dankest memes?\n", "image": "covid_memes_18.png", "hero": [], "villain": [], "victim": [], "other": ["bernie sanders", "elizabeth warren"]}

<b>Convert to </b>

covid_0	train_images/covid_memes_18.png	bernie sanders	3	Bernie or Elizabeth? Be informed.Compare them on the issues that matter. Issue: Who makes the dankest memes? <br/>
covid_0	train_images/covid_memes_18.png	elizabeth warren	3	Bernie or Elizabeth? Be informed.Compare them on the issues that matter. Issue: Who makes the dankest memes? <br/>

<b>Special Notes</b>
1. Note that due to the security purpose , only 50 samples are added to the data/ folder for skipping data conversion in model training</b> 

2. To run image related models,  images must be needed in both train_images and test_images folder</b>



## Experiments 

We have used six types of combinations to design the relationship between entity and meme using block fusion and attention mechanism. As additional experiments, we used text data augmentation. 

### Args
<b>train_tsv_file_path</b> - tsv file path of combined train data <br/>
<b>dev_tsv_file_path</b> - tsv file path of combined test data <br/>
<b>test_tsv_path</b> - tsv file path of text data with gold label <br/>
<b>checkpoint_path</b> - checkpoint path [*.pt] <br/>
<b>result_path</b> - result path [*.json] <br/>
<b>nepoch</b> - no of epochs <br/>
<b>nsamples</b> - number of samples [-1 for all samples, n-for specific number of samples] <br/>
<b>batch_sz</b> - batch size <br/>
<b>bert_mode</b>l - bert model type [bert-base-uncased,bert-base-cased] <br/>


### Entity and OCR_Text ->block fusion
python src/model_entity_text.py  --batch_sz 2 --bert_model bert-base-uncased --train_file data/combined_train.tsv  --dev_file data/combined_dev.tsv --test_file data/test_gold.tsv --nepochs 2 --checkpoint_path models/entity_text.pt --result_path results/entity_text.jsonl --nsamples 50



### Entity and Image ->block fusion
python src/model_entity_image.py  --batch_sz 2 --bert_model bert-base-uncased --train_file data/combined_train.tsv  --dev_file data/combined_dev.tsv --test_file data/test_gold.tsv --nepochs 2 --checkpoint_path models/entity_image.pt --result_path results/entity_image.jsonl --nsamples 50

### Entity and [Text, Image] ->block fusion
python src/model_entity_text_image.py  --batch_sz 2 --bert_model bert-base-uncased --train_file data/combined_train.tsv  --dev_file data/combined_dev.tsv --test_file data/test_gold.tsv --nepochs 2 --checkpoint_path models/entity_text_image.pt --result_path results/entity_text_image.jsonl --nsamples 50

### Entity and Attention(Image,Entity) ->block fusion
python src/model_attention_image.py  --batch_sz 2 --bert_model bert-base-uncased --train_file data/combined_train.tsv  --dev_file data/combined_dev.tsv --test_file data/test_gold.tsv --nepochs 2 --checkpoint_path models/entity_image_attention.pt --result_path results/entity_image_attention.jsonl --nsamples 50

### Entity and Attention(Text,Entity) ->block fusion
python src/model_attention_text.py  --batch_sz 2 --bert_model bert-base-uncased --train_file data/combined_train.tsv  --dev_file data/combined_dev.tsv --test_file data/test_gold.tsv --nepochs 2 --checkpoint_path models/entity_text_attention.pt --result_path results/entity_text_attention.jsonl --nsamples 50


### Entity and Attention([Text,Image], Entity) ->block fusion
python src/model_entity_text_image_attn.py  --batch_sz 2 --bert_model bert-base-uncased --train_file data/combined_train.tsv  --dev_file data/combined_dev.tsv --test_file data/test_gold.tsv --nepochs 2 --checkpoint_path models/entity_text_image_attention.pt --result_path results/entity_text_image_attention.jsonl --nsamples 50



