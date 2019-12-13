import models
import AudioDataset
import train_models



model3 = Net_3()
model3.to(device)
model3.apply(init_weights)

model5 = Net_5()
model5.to(device)
model5.apply(init_weights)

model11 = Net_11()
model11.to(device)
model11.apply(init_weights)

model18 = Net_18()
model18.to(device)
model18.apply(init_weights)

modelRes = Net()
modelRes.to(device)
modelRes.apply(init_weights)


models_name = ['M3','M5','M11','M18','M34-Res']
models = [model3,model5,model11,model18,modelRes]
#wandb.init(project="audio_classifier_task-")
for name,model in zip(models_name,models):
  #wandb.watch(model,log = 'all')
  #model.train()
  print('Model : {}'.format(name))
  train_test_with_CrossValidation(file_path,audio_paths,model,criterion,epochs)

  #print(model)
