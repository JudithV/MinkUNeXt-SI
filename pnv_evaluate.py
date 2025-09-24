# Judith Vilella Cantos. Miguel Hernández University of Elche.
# Evaluation using PointNetVLAD evaluation protocol and test sets
# Evaluation code adapted from PointNetVlad code: https://github.com/mikacuy/pointnetvlad
# In this modification, evaluation with a sequence from the Kitti dataset is also considered, as exposed in our work.

from sklearn.neighbors import KDTree
import numpy as np
import pickle
import os
import torch
import MinkowskiEngine as ME
import tqdm
from datasets.pointnetvlad.pnv_raw import PNVPointCloudLoader
from model.minkunext import model
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import PARAMS 
from datasets.quantization import quantizer


def evaluate(model, device, log: bool = False, show_progress: bool = False):
    # Run evaluation on all eval datasets

    eval_database_files = ['minkloc_nclt_evaluation_database.pickle']

    eval_query_files = ['minkloc_nclt_evaluation_query.pickle']

    assert len(eval_database_files) == len(eval_query_files)

    stats = {}
    for database_file, query_file in zip(eval_database_files, eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(PARAMS.dataset_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(PARAMS.dataset_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)

        temp = evaluate_dataset(model, device, database_sets, query_sets, log=log, show_progress=show_progress)
        stats[location_name] = temp

    return stats


def evaluate_dataset(model, device, database_sets, query_sets, log: bool = False,
                     show_progress: bool = False):
    # Run evaluation on a single dataset
    recall = np.zeros(25) # 5, 10, 25
    count = 0
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()

    for set in tqdm.tqdm(database_sets, disable=not show_progress, desc='Computing database embeddings'):
        database_embeddings.append(get_latent_vectors(model, set, device))

    for set in tqdm.tqdm(query_sets, disable=not show_progress, desc='Computing query embeddings'):
        query_embeddings.append(get_latent_vectors(model, set, device))
   
    for i in range(len(query_sets)):
        for j in range(len(query_sets)):
            if i == j: 
                continue
            pair_recall, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                               database_sets, log=log)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)

    ave_recall = recall / count
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall}
    return stats


def get_latent_vectors(model, set, device):
    # Adapted from original PointNetVLAD code

    if PARAMS.debug:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    pc_loader = PNVPointCloudLoader(device)

    model.eval()
    embeddings = None
    path = "/media/arvc/HDD4TB1/Judith/MinkUNeXt-main/KittiDataset"
    for i, elem_ndx in enumerate(set):
        pc_file_path = os.path.join(PARAMS.dataset_folder + "/", set[elem_ndx]["query"]) # KITTI: "query_velo"; otros: "query" .replace('.bin', '.csv')
        #pc_file_path = os.path.join(path + "/", set[elem_ndx]["query_velo"])
        pc = pc_loader(pc_file_path)
        cloud = torch.tensor(pc['cloud'], dtype=torch.float)
        reflec = torch.tensor(pc['reflec'], dtype=torch.float)
        """print("Cloud: ", cloud)
        print("Intensity: ", reflec)"""
        embedding = compute_embedding(model, cloud, reflec, device)
        if embeddings is None:
            print("Embeddings is NONE")
            embeddings = np.zeros((len(set), embedding.shape[1]), dtype=embedding.dtype)
        embeddings[i] = embedding

    return embeddings

def redondeo_personalizado(numero, umbral=0.5):
    entero = int(numero)
    decimal = numero - entero
    if decimal >= umbral:
        return entero + 1
    else:
        return entero


def compute_embedding(model, pc, reflec, device):
    coords, _ = quantizer(pc)

    with torch.no_grad():
        reflec = reflec.reshape([-1, 1])
        quatizated = ME.utils.sparse_quantize(coordinates=pc.contiguous(), features=reflec,
                                                  quantization_size=PARAMS.quantization_size)
        coords = quatizated[0]

        bcoords = ME.utils.batched_coordinates([coords]).to(device)
        """ r_2 = []
        for r in reflec:
            if r == 0:
                r_2.append(0)
            else:
                r_2.append(1)"""
        # Crear tensor de unos con el mismo número de filas que bcoords
        feats_unos = torch.ones((bcoords.shape[0], 1), dtype=torch.float32)
        reflec = quatizated[1]
        feats_composed = []
        """for i, f in enumerate(reflec.numpy()):
            feats_composed.append(redondeo_personalizado(f, 0.1))
         if f <= 0.1:
                feats_composed.append([1,1,0,0]) # Reflectividad baja
            elif 0.1 < f < 0.5:
                feats_composed.append([1,0,1,0]) # Reflectividad media
            else:
                feats_composed.append([1,0,0,1]) # Reflectividad alta
        feats_composed = torch.tensor(feats_composed, dtype=torch.float32)
        #feats_composed = torch.cat(feats_composed, dim=0)
        feats_composed = feats_composed.unsqueeze(1)"""
        # Verificar si reflec es una lista y concatenar si es necesario
        if isinstance(reflec, list):
            reflec = torch.cat(reflec, dim=0)
        # Asegurarse de que reflec tiene dos dimensiones
        if reflec.dim() == 1:
            reflec = reflec.unsqueeze(1)

        feats_r = reflec

        # Verificar que las dimensiones coinciden antes de la concatenación
        if feats_unos.shape[0] != feats_r.shape[0]:
            raise ValueError(f"Las dimensiones no coinciden: feats_unos tiene {feats_unos.shape[0]} filas, pero reflec tiene {feats_r.shape[0]} filas.")

        # Concatenar feats_unos y feats_r a lo largo de la dimensión 1
        feats_doble = torch.cat((feats_unos, feats_r), dim=1)
        feats_uno_y_r = feats_unos + feats_r

        # Preparar batch para el modelo
        if PARAMS.use_intensity:
            batch = {'coords': bcoords.to(device), 'features': feats_r.to(device)}
        else:
            batch = {'coords': bcoords.to(device), 'features': feats_unos.to(device)}

        # Compute global descriptor
        y = model(batch)
        embedding = y['global'].detach().cpu().numpy()

    return embedding

def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    # When embeddings are normalized, using Euclidean distance gives the same
    # nearest neighbour search results as using cosine distance
    database_nbrs = KDTree(database_output)

    num_neighbors = 25 # 5, 10, 25
    recall = [0] * num_neighbors

    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output)/100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]    # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1

        # Find nearest neightbours
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        if log:
            # Log false positives (returned as the first element) for Oxford dataset
            # Check if there's a false positive returned as the first element
            if query_details['query'][:6] == 'oxford' and indices[0][0] not in true_neighbors:
                fp_ndx = indices[0][0]
                fp = database_sets[m][fp_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                fp_emb_dist = distances[0, 0]  # Distance in embedding space
                fp_world_dist = np.sqrt((query_details['northing'] - fp['northing']) ** 2 +
                                        (query_details['easting'] - fp['easting']) ** 2)
                # Find the first true positive
                tp = None
                for k in range(len(indices[0])):
                    if indices[0][k] in true_neighbors:
                        closest_pos_ndx = indices[0][k]
                        tp = database_sets[m][closest_pos_ndx]  # Database element: {'query': path, 'northing': , 'easting': }
                        tp_emb_dist = distances[0][k]
                        tp_world_dist = np.sqrt((query_details['northing'] - tp['northing']) ** 2 +
                                                (query_details['easting'] - tp['easting']) ** 2)
                        break

                with open("log_fp.txt", "a") as f:
                    s = "{}, {}, {:0.2f}, {:0.2f}".format(query_details['query'], fp['query'], fp_emb_dist, fp_world_dist)
                    if tp is None:
                        s += ', 0, 0, 0\n'
                    else:
                        s += ', {}, {:0.2f}, {:0.2f}\n'.format(tp['query'], tp_emb_dist, tp_world_dist)
                    f.write(s)

            if query_details['query'][:6] == 'oxford':
                # Save details of 5 best matches for later visualization for 1% of queries
                s = f"{query_details['query']}, {query_details['northing']}, {query_details['easting']}"
                for k in range(min(len(indices[0]), 5)):
                    is_match = indices[0][k] in true_neighbors
                    e_ndx = indices[0][k]
                    e = database_sets[m][e_ndx]     # Database element: {'query': path, 'northing': , 'easting': }
                    e_emb_dist = distances[0][k]
                    world_dist = np.sqrt((query_details['northing'] - e['northing']) ** 2 +
                                         (query_details['easting'] - e['easting']) ** 2)
                    s += f", {e['query']}, {e_emb_dist:0.2f}, , {world_dist:0.2f}, {1 if is_match else 0}, "
                s += '\n'
                out_file_name = "log_search_results.txt"
                with open(out_file_name, "a") as f:
                    f.write(s)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved/float(num_evaluated))*100
    recall = (np.cumsum(recall)/float(num_evaluated))*100
    return recall, one_percent_recall


def print_eval_stats(stats):
    for database_name in stats:
        print('Dataset: {}'.format(database_name))
        t = 'Avg. top 1% recall: {:.2f}   Avg. recall @N:'
        print(t.format(stats[database_name]['ave_one_percent_recall']))
        print(stats[database_name]['ave_recall'])


def pnv_write_eval_stats(file_name, prefix, stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in stats:
            ave_1p_recall = stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


if __name__ == "__main__":
  
    if torch.cuda.is_available():
        device = PARAMS.cuda_device
        torch.cuda.set_device(device)
    else:
        device = "cpu"
    print('Device: {}'.format(device))
    
    model.load_state_dict(torch.load(PARAMS.weights_path, map_location=device))

    model.to(device)

    stats = evaluate(model, device)
    print_eval_stats(stats)

    # Save results to the text file

    model_name = os.path.split(PARAMS.weights_path)[1]
    model_name = os.path.splitext(model_name)[0]
    prefix = "{}, {}".format(PARAMS.protocol, model_name)
    pnv_write_eval_stats("results.txt", prefix, stats)


