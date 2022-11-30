# Retrieving Multimodal Prompts for Medical Visual Question Answering

This repository includes the source code for T5vision, a model which optionally integrates visual features or retrieved information into T5 via prompting.

## How to use
### Setup the environment
TODO: provide instructions
### Obtain the data
To reproduce our results, first clone this repository and download the SLAKE dataset from [their website](https://www.med-vqa.com/slake/). Also obtain the image folder, trainset.json, and testset.json files from the data folder at [this github repository](https://github.com/Awenbocc/med-vqa). Once the data has been obtained, organize the data into this directory structure (note, VQA_RAD json files should also be renamed to train.json, test.json):

```
data
|----SLAKE
     |----train.json
     |----validate.json
     |----test.json
     |----imgs
          |----xmlab0
          |----xmlab1
                ⋮
|----VQA_RAD
    |----train.json
    |----test.json
    |----imgs
          |----synpic100132.jpg
          |----synpic100176.jpg
                ⋮
```
### Setup the config file
We provide a sample config file in the config folder. You may edit the various entries within it to customize the experiment.
### Run main.py
To train a model, execute the following command:
```
python main.py --train --config <config_file_name>
```
To test a model, use the following:
```
python main.py --test --config <config_file_name>
```


