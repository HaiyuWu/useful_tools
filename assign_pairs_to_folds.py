import numpy as np
from os import path, makedirs
from collections import defaultdict
import argparse


def order_by_identity(args):
    gen, imp = np.genfromtxt(args.genuine_file, dtype=str), np.genfromtxt(args.impostor_file, dtype=str)
    chunk_size = args.fold_size // 2
    all_pairs = defaultdict(lambda: {'gen': [], 'imp': []})
    for gen_pair in gen:
        identity = path.split(gen_pair[0])[1].split("_")[0]
        all_pairs[identity]['gen'].append(gen_pair)

    for imp_pair in imp:
        identity = path.split(imp_pair[0])[1].split("_")[0]
        all_pairs[identity]['imp'].append(imp_pair)

    # Initialize lists for genuine and imposter
    gen_list = [[] for _ in range(args.fold_num)]
    imp_list = [[] for _ in range(args.fold_num)]
    identity_in_sublists = defaultdict(set)
    leftover_pairs = {'gen': [], 'imp': []}

    # First pass: fill up sublists without overlapping identities
    for identity, pairs in all_pairs.items():
        for i in range(args.fold_num):
            if (len(gen_list[i]) + len(pairs['gen'])) <= chunk_size and (len(imp_list[i]) + len(pairs['imp'])) <= chunk_size:
                gen_list[i].extend(pairs['gen'])
                imp_list[i].extend(pairs['imp'])
                identity_in_sublists[identity].add(i)
                # Remove the added pairs to avoid re-adding them later
                del pairs['gen']
                del pairs['imp']
                break
        else:
            # If an identity could not be fully placed, add to leftover pairs
            leftover_pairs['gen'].extend(pairs['gen'])
            leftover_pairs['imp'].extend(pairs['imp'])

    # Second pass: distribute leftover pairs
    for category in ['gen', 'imp']:
        for pair in leftover_pairs[category]:
            identity = path.split(pair[0])[1].split("_")[0]
            for i in range(args.fold_num):
                sublist = gen_list[i] if category == 'gen' else imp_list[i]
                if len(sublist) < chunk_size:
                    sublist.append(pair)
                    identity_in_sublists[identity].add(i)
                    break

    # Report the number of pairs in gen_list[i] and imp_list[i]
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
