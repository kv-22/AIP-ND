import argparse
import m
import u
import json

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k', default=1, type=int)
    parser.add_argument('--category_names', default='/home/workspace/ImageClassifier/cat_to_name.json')
    parser.add_argument('--gpu', action='store_true')
    return parser.parse_args()
    

def main():
    
    in_arg = get_input_args()
    img_path = in_arg.img_path
    topn = in_arg.top_k
    category_names = in_arg.category_names
    device = 'cuda' if in_arg.gpu else 'cpu'
    with open(category_names, 'r') as f:
        names = json.load(f)

    model = m.load_checkpoint(in_arg.checkpoint, device)
    probs, classes = u.predict(img_path, model, topn, device)
    x = []
    for c in classes:
        for key, value in names.items():
            if str(c) == key:
                x.append(value)

    sorted_p_l = sorted(zip(probs, x), reverse=True)  # Sort in descending order
    probabilities, names = zip(*sorted_p_l)
    print("Probabilities: ", probabilities)
    print("Flower names: ", names)
    
if __name__ == "__main__":
    main()
