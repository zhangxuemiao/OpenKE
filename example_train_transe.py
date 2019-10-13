import os

import config
import models

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Input training files from benchmarks/FB15K/ folder.
con = config.Config()
# True: Input test files from the same folder.
con.set_in_path(os.path.join(con.PROJECT_PATH, 'benchmarks/FB15K/'))
con.set_test_link_prediction(True)
con.set_test_triple_classification(True)
con.set_work_threads(8)
con.set_train_times(10)
con.set_nbatches(100)
con.set_alpha(0.001)
con.set_margin(1.0)
con.set_bern(0)
con.set_dimension(100)
con.set_ent_neg_rate(1)
con.set_rel_neg_rate(0)
con.set_opt_method("SGD")

model_res_dir = os.path.join(con.PROJECT_PATH, 'res/transe/')
con.make_dir(model_res_dir)

# Models will be exported via tf.Saver() automatically.
con.set_export_files(os.path.join(model_res_dir, 'model.vec.tf'), 0)
# Model parameters will be exported to json files automatically.
con.set_out_files(os.path.join(model_res_dir, 'embedding.vec.json'))
# Initialize experimental settings.
con.init()
# Set the knowledge embedding model
con.set_model(models.TransE)
# Train the model.
con.run()
# To test models after training needs "set_test_flag(True)".
con.test()
con.show_link_prediction(2, 1)
con.show_triple_classification(2, 1, 3)
