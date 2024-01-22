import argparse
import m 

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir') 
    parser.add_argument('--save_dir', default = '/home/workspace/ImageClassifier/')
    parser.add_argument('--learning_rate', default = 0.003, type=float)
    parser.add_argument('--hidden_units', default = 600, type=int)
    parser.add_argument('--epochs', default = 5, type=int)
    parser.add_argument('--arch', default = 'vgg16' )
    parser.add_argument('--gpu', action='store_true' )
 

    return parser.parse_args()

def main():
    in_arg = get_input_args()
    data_dir = in_arg.dir
    arch = in_arg.arch
    if arch not in ['vgg16', 'vgg13']:
        print("Use vgg16 or vgg13.")
        return
    hidden_units = in_arg.hidden_units
    device = 'cuda' if in_arg.gpu else 'cpu'
    n_epochs = in_arg.epochs
    learning_rate = in_arg.learning_rate
    save_dir = in_arg.save_dir
    image_datasets, dataloaders = m.transformations(data_dir)
    model = m.create_model(hidden_units, arch)
    model, optimizer = m.train_model(model, device, n_epochs, learning_rate, dataloaders)
    m.create_checkpoint(save_dir, model, image_datasets, optimizer, n_epochs, learning_rate, arch)
    
    
    
    
if __name__ == "__main__":
    main()

