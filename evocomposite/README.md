# Evaluate the performance of Variable-length Composite Adversarial Perturbations


## Usage
### Evaluate robust accuracy / attack success rate of the model
```shell
python eval_VCSP.py \
       --arch resnet50 --checkpoint PATH_TO_MODEL \
       --dataset DATASET_NAME --dataset-path DATASET_PATH --input-normalized \
       --message MESSAGE_TO_PRINT_IN_CSV \
       --batch_size BATCH_SIZE --output RESULT.csv \
       "NoAttack()" \
       "CompositeAttack(model, enabled_attack=(0,), order_schedule='fixed', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(1,), order_schedule='fixed', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(2,), order_schedule='fixed', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(3,), order_schedule='fixed', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(4,), order_schedule='fixed', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(5,), order_schedule='fixed', inner_iter_num=20)" \
       "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='random', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,1,5), order_schedule='scheduled', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='random', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(3,4,5), order_schedule='scheduled', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='random', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,2,5), order_schedule='scheduled', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='random', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,1,2,3,4), order_schedule='scheduled', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,1,2,3,4,5), order_schedule='random', inner_iter_num=10)" \
       "CompositeAttack(model, enabled_attack=(0,1,2,3,4,5), order_schedule='scheduled', inner_iter_num=10)" \
       "AutoLinfAttack(model, 'cifar', bound=8/255)"
```
