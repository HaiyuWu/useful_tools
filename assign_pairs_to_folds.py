import numpy as np
from os import path, makedirs
from collections import defaultdict
import argparse


def order_by_identity(args):
    gen, imp = np.genfromtxt(args.genuine_file, dtype=str), np.genfromtxt(args.impostor_file, dtype=str)

    all_pairs = defaultdict(lambda: {'gen': [], 'imp': []})
    assigned_gen_pairs = set()
    assigned_imp_pairs = set()

    for gen_pair in gen:
        identity = path.basename(gen_pair[0]).split("_")[0]
        all_pairs[identity]['gen'].append(tuple(gen_pair))

    for imp_pair in imp:
        identities = [path.basename(imp_pair[i]).split("_")[0] for i in range(2)]
        imp_pair_tuple = tuple(imp_pair)
        all_pairs[identities[0]]['imp'].append(imp_pair_tuple)
        if identities[0] != identities[1]:
            all_pairs[identities[1]]['imp'].append(imp_pair_tuple)

    gen_list = [[] for _ in range(args.fold_num)]
    imp_list = [[] for _ in range(args.fold_num)]

    # Distribute pairs evenly across folds
    for fold in range(10):
        for identity, pairs in all_pairs.items():
            for gen_pair in pairs['gen']:
                if len(gen_list[fold]) < args.fold_size and gen_pair not in assigned_gen_pairs:
                    gen_list[fold].append(gen_pair)
                    assigned_gen_pairs.add(gen_pair)

            for imp_pair in pairs['imp']:
                if len(imp_list[fold]) < args.fold_size and imp_pair not in assigned_imp_pairs:
                    imp_list[fold].append(imp_pair)
                    assigned_imp_pairs.add(imp_pair)

    # Reporting sizes for debugging purposes
    for i in range(args.fold_num):
        print(f'gen_list[{i}] size: {len(gen_list[i])}')
        print(f'imp_list[{i}] size: {len(imp_list[i])}')

    return gen_list, imp_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "This is for reduce the identity overlapping between test folds")
    parser.add_argument("--genuine_file", "-gen", help="genuine file", type=str)
    parser.add_argument("--impostor_file", "-imp", help="impostor file", type=str)
    parser.add_argument("--dest_path", "-d",
                        help="destination to save the results",
                        type=str,
                        default="./final_version")
    parser.add_argument("--name", "-name", help="file name", type=str, default="pairs")
    parser.add_argument("--fold_size", "-size", help="the size of each fold", type=int, default=600)
    parser.add_argument("--fold_num", "-num", help="the number of folds", type=int, default=10)

    args = parser.parse_args()

    gen_list, imp_list = order_by_identity(args)

    final_pairs = []
    for gen_pairs, imp_pairs in zip(gen_list, imp_list):
        final_pairs += gen_pairs
        final_pairs += imp_pairs
    if not path.join(args.dest_path):
        makedirs(args.dest_path)
    np.savetxt(f"{args.dest_path}/{args.name}.txt", final_pairs, fmt="%s")
