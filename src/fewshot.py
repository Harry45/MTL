"""
Description: Networks related to the few shot learning part.

Author: Arrykrishna Mootoovaloo
Date: August 2022
Email: arrykrish@gmail.com/a.mootoovaloo17@imperial.ac.uk/arrykrishna.mootoovaloo@physics.ox.ac.uk
Project: One/Few-Shot Learning for Galaxy Zoo
"""

import os
import shutil
import random
import subprocess
import warnings
import datetime
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

# our scripts and functions
import settings as st
import utils.helpers as hp
from src.network import MultiLabelNet
from src.dataset import FSdataset


warnings.filterwarnings("ignore")

DATE = datetime.datetime.now()
TODAY = str(DATE.year) + '-' + str(DATE.month) + '-' + str(DATE.day)
TIME = str(DATE.hour) + ':' + str(DATE.minute)


def select_objects(dataframe: pd.DataFrame, columns: list,
                   threshold: float = 0.90, nobjects: int = 20, save: bool = False) -> dict:
    """Select objects from the pandas dataframe given a threshold for the
    volunteers' votes

    Args:
        dataframe (pd.DataFrame): a dataframe with the volunteers votes
        columns (list): a list of the columns, for which the threshold will be
        applied
        threshold (float, optional): the threshold value. Defaults to 0.90.
        nobjects (int, optional): the number of objects to be selected. Defaults to 20.
        save (bool, optional): Option to save the files. Defaults to False.

    Returns:
        dict: Dictionary of the selected objects
    """
    dataframe.rename(st.MAPPING, axis=1, inplace=True)

    objects = dict()

    for col in columns:
        objects[col] = dataframe[dataframe[col] > threshold]

        print(f'{col} has {len(objects[col])} objects')

        assert len(objects[col]) > nobjects, "too many objects selected"

        randint = random.sample(range(0, len(objects[col])), nobjects)

        objects[col] = objects[col].iloc[randint]

        # reset pandas index
        objects[col].reset_index(drop=True, inplace=True)

    if save:
        hp.save_pickle(objects, 'fewshot', 'attributes')

    return objects


def copy_image_fewshot(nobjects: int = 50, threshold: float = 0.90):
    """Copy examples of similar objects from Mike's folder to the fewshot/images/ directory.

    Args:
        nobjects (int): number of objects to copy. Defaults to 50.
        threshold (float): the threshold to apply to the volunteers' votes. Defaults to 0.90.
    """

    # remove existing images if they exist
    dirpath = "fewshot/images/"
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

    os.makedirs(dirpath, exist_ok=True)

    # read the dataframe
    dataframe = hp.read_parquet(st.DATA_DIR, 'descriptions/dr5_votes')

    # we need to show the network images that it has never seen before (robust analysis)
    testpoints = hp.load_csv(st.DATA_DIR, 'ml/test')
    subdata = dataframe[dataframe.iauname.isin(testpoints.iauname)]

    # select the objects
    objects = select_objects(subdata, st.FS_COLS, threshold, nobjects=nobjects, save=True)

    for col in st.FS_COLS:

        # select the common objects
        obj = objects[col]

        # name of the folder
        colname = col.replace(" ", "-").replace("(", "").replace(")", "")

        # folder where we want to save the objects
        folder = 'fewshot/images/' + colname

        # make the folder
        os.makedirs(folder, exist_ok=True)

        for i in range(nobjects):
            path = st.DECALS + '/' + obj.png_loc.iloc[i]
            subprocess.run(["cp", path, folder], capture_output=True, text=True)


def targets_support_query(nshot: int, save: bool = False, train: bool = True):
    """Here we assume we have selectively chosen the query images to report an
    accuracy measure.

    Args:
        nshot (int): number of shots we are using. Defaults to 10.
        save (bool, optional): Option to save the files. Defaults to False.
        train (bool, optional): Will choose either train or validate.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two pandas dataframes,
        query and support.
    """

    if train:
        folder = 'train'
    else:
        folder = 'validate'

    query_main = os.listdir(f'fewshot/{folder}/query/')

    query_dataframe = list()
    support_dataframe = list()

    for objtype in st.FS_CLASSES:

        # list the images in the main folder
        main = os.listdir(f'fewshot/images/{objtype}/')

        # list the images in the support folder
        support = os.listdir(f'fewshot/{folder}/{str(nshot)}-shots/{objtype}/')

        # get the list of query images
        query = list(set(main).intersection(query_main))

        # number of objects
        nquery = len(query)
        nsupport = len(support)

        # create dataframes to store the labels
        df_query = pd.DataFrame()
        df_query['Objects'] = query
        df_query['Labels'] = [objtype] * nquery

        df_support = pd.DataFrame()
        df_support['Objects'] = support
        df_support['Labels'] = [objtype] * nsupport

        # store the different dataframes
        query_dataframe.append(df_query)
        support_dataframe.append(df_support)

    query_dataframe = pd.concat(query_dataframe)
    support_dataframe = pd.concat(support_dataframe)

    if save:
        hp.save_pd_csv(query_dataframe, 'fewshot', f'{folder}/query_{str(nshot)}')
        hp.save_pd_csv(support_dataframe, 'fewshot', f'{folder}/support_{str(nshot)}')

    return query_dataframe, support_dataframe


def copy_query_images(nshot: int = 10, save: bool = False, train: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Copy the query images to the fewshot/query/ directory.

    Args:
        nshot (int): number of shots we are using. Defaults to 10.
        save (bool, optional): Option to save the files. Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two pandas dataframes,
        query and support.
    """

    dirpath = 'fewshot/query/'

    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

    os.makedirs(dirpath, exist_ok=True)

    query_dataframe = list()
    support_dataframe = list()

    for objtype in st.FS_CLASSES:

        # list the images in the main folder
        main = os.listdir(f'fewshot/images/{objtype}/')

        # list the images in the support folder
        support = os.listdir(f'fewshot/{str(nshot)}-shots/{objtype}/')

        # create the list of query objects
        query = list(set(main) ^ set(support))

        # number of objects
        nquery = len(query)
        nsupport = len(support)

        # create dataframes to store the labels
        df_query = pd.DataFrame()
        df_query['Objects'] = query
        df_query['Labels'] = [objtype] * nquery

        df_support = pd.DataFrame()
        df_support['Objects'] = support
        df_support['Labels'] = [objtype] * nsupport

        # get the full paths for the query objects and copy them
        paths = [f'fewshot/images/{objtype}/' + query[i] for i in range(nquery)]

        for i in range(nquery):
            subprocess.run(["cp", paths[i], dirpath], capture_output=True, text=True, check=True)

        # store the different dataframes
        query_dataframe.append(df_query)
        support_dataframe.append(df_support)

    query_dataframe = pd.concat(query_dataframe)
    support_dataframe = pd.concat(support_dataframe)

    if save:
        hp.save_pd_csv(query_dataframe, 'fewshot', f'query_{str(nshot)}')
        hp.save_pd_csv(support_dataframe, 'fewshot', f'support_{str(nshot)}')

    return query_dataframe, support_dataframe


def ml_backbone(modelname: str):
    """Returns the model (backbone) which outputs the embeddings for the
    image.This function is for the multilabel case only.

    Args:
        modelname (str): name of the trained model, for example,
        ml-models-2022-5-25/resnet_18_multilabel_29.pth
    """

    # full path to the model
    fullpath = os.path.join(st.MODEL_PATH, modelname)

    # load the model
    loaded_model = torch.load(fullpath, map_location=st.DEVICE)

    # create the model again
    model = MultiLabelNet(backbone="resnet18")
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(loaded_model)
    model.to(st.DEVICE)
    model.eval()

    # the backbone
    chopped_layer = nn.Sequential(list(model.children())[0].backbone)

    return chopped_layer


def ml_feature_extractor(model: torch.nn.modules, dataloaders: dict, save: bool, train: bool = True) -> torch.Tensor:
    """Extract the embeddings from the trained model given an image.

    Args:
        model (torch.nn.modules): the backbone.
        dataloaders (dict): the dataloaders for the support set.
        save (bool): whether to save the embeddings or not.
        train (bool): if True, the data from the training set will be used.
    Returns:
        torch.Tensor: the feature vector
    """
    if train:
        folder = 'train'
    else:
        folder = 'validate'

    # create an empty dictionary to store the features
    vectors = dict()

    # an empty list to store the mean embedding for each class
    vectors_mean = dict()

    for col in st.FS_CLASSES:

        # get the dataloader for the current class
        loader = dataloaders[col]

        # create an empty list to store the features for the current class
        vectors[col] = list()

        for i in range(st.NSHOTS):
            img = loader.dataset[i].view(1, 1, st.IMG_SIZE[-1], st.IMG_SIZE[-1]).to(st.DEVICE)
            vec = model(img).data.to('cpu').view(-1)
            vectors[col].append(vec)

        # convert the list to a tensor (10 x 1000) - each row is the embedding for a single image
        vectors[col] = torch.vstack(vectors[col])
        vectors_mean[col] = vectors[col].mean(dim=0)

    # number of shots
    nshots = len(dataloaders[col].dataset)

    if save:
        hp.save_pickle(vectors, "fewshot", f"{folder}/vectors_{str(nshots)}")
        hp.save_pickle(vectors_mean, "fewshot", f"{folder}/vectors_mean_{str(nshots)}")

    return vectors, vectors_mean


def distance_support_query(modelname: str, nshot: int, save: bool, train: bool = False) -> pd.DataFrame:
    """Calculates the L1 distance (Manhattan or Taxicab) between the query and
    centroid of the support sets.

    Args:
        modelname (str): The model to use to create the embedding vector.
        nshot (int): The number of examples in the support sets.
        save (bool): Whether to save the data or not.
        train (bool): If we want to use the query set from the training set or
        validate set.

    Returns:
        pd.DataFrame: The dataframe consisting of the true labels and the
        predicted labels.
    """

    if train:
        folder = 'train'
    else:
        folder = 'validate'

    # mean vector computed and stored
    vectors_mean = hp.load_pickle('fewshot', f'{folder}/vectors_mean_{str(nshot)}')

    # the model loaded
    model = ml_backbone(modelname)

    # the dataloader for the query set
    querydata = FSdataset(support=False, train=train)
    queryloader = DataLoader(dataset=querydata, batch_size=1, shuffle=False)
    nquery = len(queryloader.dataset)

    # create an empty list to store the normalised vectors for the support sets.
    class_support = list()

    for key in st.FS_CLASSES:
        v_norm = F.normalize(vectors_mean[key].view(1, -1))
        class_support.append(v_norm)

    # convert to a tensor (this is of size 4 x 1000)
    class_support = torch.cat(class_support, dim=0)

    print(f'The shape of the supports embeddings is {class_support.shape[0]} x {class_support.shape[1]}')

    distance_l1 = dict()

    for idx in range(nquery):
        img = queryloader.dataset[idx].view(1, 1, st.IMG_SIZE[-1], st.IMG_SIZE[-1]).to(st.DEVICE)
        print(img)
        vec = model(img).data.to('cpu').view(-1)

        # normalise vector
        vec_norm = F.normalize(vec.view(1, -1))

        # pairwise distance
        dist = torch.cdist(vec_norm, class_support, p=1)

        # name of the file
        name = os.path.split(queryloader.dataset.fnames[idx])[-1]
        distance_l1[name] = dist.view(-1).data.numpy()

    distance_l1 = pd.DataFrame(distance_l1).T

    # rename the columns
    distance_l1.columns = st.FS_CLASSES

    # find the column name with the minimum distance
    labels_pred = distance_l1.idxmin(axis="columns")
    labels_pred = labels_pred.reset_index(level=0)

    # rename the columns
    labels_pred.columns = ['Objects', 'Labels']

    # load the true labels and merge them with the predicted labels
    truth = hp.load_csv('fewshot', f'{folder}/query_{str(nshot)}')
    combined = pd.merge(truth, labels_pred, on='Objects', how='outer')

    # rename the columns for the combined dataframe
    combined.columns = ['Objects', 'True Labels', 'Predicted Labels']

    if save:
        hp.save_pd_csv(combined, 'fewshot', f'{folder}/nearest_neighbour_{str(nshot)}')

    return combined


def generate_labels_fewshot(nshot: int, save: bool, train: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a csv file with the class labels for the support and query set.

    Args:
        nshot (int): number of examples in the support set.
        save (bool): if True, the files will be saved.
        train (bool): if True, the training set will be generated.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame] : dataframes for the query and support
       sets with the following columns [Objects, Labels, Targets] in the dataframes.
    """

    if train:
        folder = 'train'
    else:
        folder = 'validate'

    # load the csv files with the object names
    labels_query = hp.load_csv('fewshot', f'{folder}/query_{str(nshot)}')
    labels_support = hp.load_csv('fewshot', f'{folder}/support_{str(nshot)}')

    # generate a column consisting of the integer labels for the query and
    # support sets
    labels_query['Targets'] = pd.factorize(labels_query['Labels'])[0]
    labels_support['Targets'] = pd.factorize(labels_support['Labels'])[0]

    if save:
        hp.save_pd_csv(labels_query, 'fewshot', f'{folder}/query_targets_{str(nshot)}')
        hp.save_pd_csv(labels_support, 'fewshot', f'{folder}/support_targets_{str(nshot)}')

    return labels_query, labels_support


def shannon_entropy(pred_logits: torch.Tensor) -> torch.Tensor:
    """Calculates the shannon entropy of the predicted logits.

    Args:
        pred_logits (torch.Tensor): The predicted logits.

    Returns:
        torch.Tensor: The shannon entropy of the predicted logits.
    """

    # calculate the probabilities
    probabilities = F.softmax(pred_logits, dim=1)

    # calculate the log-probabilities
    log_probabilities = F.log_softmax(pred_logits, dim=1)

    # calculate the loss (regularisation term)
    loss = - (probabilities * log_probabilities).sum(dim=1).mean()

    return loss


def normalise_weights(folder: str, fname: str) -> torch.Tensor:
    """Given the embedding vectors for the support sets, normalises the vectors

    Args:
        folder (str): folder where the mean vector is stored
        fname (str): name of the mean vector file

    Returns:
        torch.Tensor: The normalised embedding vectors of size Nclass (4) x 1000.
    """

    weightmatrix = hp.load_pickle(folder, fname)

    weights = []

    for item in weightmatrix:
        weights.append(weightmatrix[item])

    weights = torch.row_stack(weights)
    weights_norm = torch.nn.functional.normalize(weights, dim=1)

    return weights_norm


def training_finetune(model: nn.Module, loaders: dict, quant: dict, save: bool) -> nn.Module:
    """Fine tune the model and adapt it to the few shot learning task.

    Args:
        model (nn.Module): the pre-trained deep learning model.
        loaders (dict): a dictionary for the support and query data.
        quant (dict): a dictionary with important quantities, for example,

        quant = {'lr': 1E-3,
        'weight_decay': 1E-5,
        'coefficient': 0.01,
        'nepochs': 150,
        'criterion': nn.CrossEntropyLoss()}

        where weight decay is a regularizer, coefficient is a regularization
        term for the Shannon Entropy calculation and so forth.

        save (bool): if True, the model will be save to the models directory

    Returns:
        nn.Module: the fine tuned model.
    """

    # choose an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=quant['lr'], weight_decay=quant['weight_decay'])

    # iterate over the whole dataset
    for epoch in range(quant['nepochs']):

        model.train()

        losses = []
        loss_support_rec = []
        loss_query_rec = []

        # calculate loss due to the support set
        for images, targets in loaders['support']:
            images, targets = map(lambda x: x.to(st.DEVICE), [images, targets])

            outputs = model(images)

            loss_support = quant['criterion'](outputs, targets.view(-1))

            optimizer.zero_grad()
            loss_support.backward()
            optimizer.step()

            loss_support_rec.append(loss_support.item())

        # calculate loss due to the query set (transductive learning)
        for images, targets in loaders['query']:
            images, targets = map(lambda x: x.to(st.DEVICE), [images, targets])

            outputs = model(images)

            loss_query = quant['coefficient'] * shannon_entropy(outputs)

            optimizer.zero_grad()
            loss_query.backward()
            optimizer.step()

            loss_query_rec.append(loss_query.item())

        # calculate the total loss
        total_loss = sum(loss_support_rec) / len(loss_support_rec) + sum(loss_query_rec) / len(loss_query_rec)

        losses.append(total_loss)

        print(f"Epoch [{epoch + 1} / {quant['nepochs']}]: Loss = {total_loss:.4f}")

    # save the values of the loss
    if save:
        hp.save_pickle(losses, 'results', f'loss_finetune_{TODAY}')
        torch.save(model.state_dict(), st.MODEL_PATH + f'finetune_{TODAY}.pth')

    return model


def finetuning_predictions(model: nn.Module, queryloader: DataLoader, nshot: int, save: bool) -> pd.DataFrame:
    """_summary_

    Args:
        model (nn.Module): _description_
        queryloader (DataLoader): _description_
        nshot (int): _description_
        save (bool):

    Returns:
        pd.DataFrame: _description_
    """

    # set model to evaluation mode
    model.eval()

    nquery = len(queryloader.dataset)

    probabilities = {}

    for i in range(nquery):

        # calculate the logits
        logits = model(queryloader.dataset[i][0].view(1, 1, 224, 224).to(st.DEVICE))

        # calculate the probabilities
        prob = F.softmax(logits.data.cpu(), dim=1)

        # name of the file
        name = queryloader.dataset.csvfile['Objects'].values[i]
        probabilities[name] = prob.view(-1).data.numpy()

    probabilities = pd.DataFrame(probabilities).T

    # rename the columns
    probabilities.columns = st.FS_CLASSES

    # find the column name with the maximum probability
    labels_pred = probabilities.idxmax(axis="columns")
    labels_pred = labels_pred.reset_index(level=0)

    # rename the columns
    labels_pred.columns = ['Objects', 'Labels']

    # load the true labels and merge them with the predicted labels
    truth = hp.load_csv('fewshot', f'query_{str(nshot)}')
    combined = pd.merge(truth, labels_pred, on='Objects', how='outer')

    # rename the columns for the combined dataframe
    combined.columns = ['Objects', 'True Labels', 'Predicted Labels']

    if save:
        hp.save_pd_csv(combined, 'fewshot', f'finetune_{str(nshot)}')

    return combined
