#%%
#1. Import packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,optimizers,losses,callbacks,applications
import numpy as np
import matplotlib.pyplot as plt
import os,datetime, pickle
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model

#%% 2. Data loading
file_path = r"C:\Users\USER\Desktop\Assesment 3\dataset"
#test_path = os.path.join(file_path,"test")
#train_path = os.path.join(file_path,"train")
#%%
BATCH_SIZE = 32
IMG_SIZE = (224,224)
file_dataset = keras.utils.image_dataset_from_directory(file_path,batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True)
#train_dataset = keras.utils.image_dataset_from_directory(train_path,batch_size=BATCH_SIZE,image_size=IMG_SIZE,shuffle=True)

#%%
#Take first batch of test data as the test dataset, the rest will be validation dataset
val_dataset = file_dataset.skip(1)
test_dataset = file_dataset.take(1)
# %%
#3. Convert the datasets into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE
pf_train = file_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = val_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)
# %%
#4. Create the data augmentation model
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))
# %%
#5. Create the input preprocessing layer
preprocess_input = applications.mobilenet_v2.preprocess_input
# %%
#6. Apply transfer learning
class_names = file_dataset.class_names
nClass = len(class_names)
#(A) Apply transfer learning to create the feature extractor
IMG_SHAPE = IMG_SIZE + (3,)
base_model = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights="imagenet")
base_model.trainable = False
# %%
#(B) Create the classifier
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(nClass,activation='softmax')
# %%
#7. Link the layers together to form the model pipeline using functional API
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x,training=False)
x = global_avg(x)
# x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)
model.summary()

plot_model(model,show_shapes=True)
# %%
#8. Compile the model
cos_decay = optimizers.schedules.CosineDecay(0.0005,50)
optimizer = optimizers.Adam(learning_rate=cos_decay)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])



# %%
#9. Evaluate the model before training
loss0,acc0 = model.evaluate(pf_test)
print("----------------Evaluation Before Training-------------------")
print("Loss = ",loss0)
print("Accuracy = ",acc0)
# %%
#10. Create tensorboard
log_dir = os.path.join(os.getcwd(),datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb_callback = TensorBoard(log_dir=log_dir)
es_callback = EarlyStopping(monitor='loss',patience=5)
# %%
#11. Model training
EPOCHS = 3
es_callback = EarlyStopping(monitor='loss',patience=5)
history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,callbacks=[tb_callback,es_callback])
# %%
#12. Follow-up training
base_model.trainable = True
for layer in base_model.layers[:200]:
    layer.trainable = False
base_model.summary()
# %%
#13. Compile the model
optimizer = optimizers.RMSprop(learning_rate=0.00001)
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])
# %%
#14. Continue training the model
fine_tune_epoch = 3
total_epoch = EPOCHS + fine_tune_epoch
history_fine = model.fit(pf_train,validation_data=pf_val,epochs=total_epoch,initial_epoch = history.epoch[-1],callbacks=[tb_callback,es_callback])
# %%
#15. Evaluate the model after training
test_loss, test_acc = model.evaluate(pf_test)
print("----------------Evaluation After Training---------------")
print("Test loss = ",test_loss)
print("Test accuracy = ",test_acc)
# %%
#16. Model deployment
image_batch, label_batch = pf_test.as_numpy_iterator().next()
y_pred = np.argmax(model.predict(image_batch),axis=1)
#Stack the label and prediction in one numpy array
label_vs_prediction = np.transpose(np.vstack((label_batch,y_pred)))
# %%
model.save('model.h5')



# %%
