import tensorflow as tf
import time
import tensorflow_hub as hub
import gradio as gr

L_M = tf.keras.models.load_model("Cupid_2_0_c.h5",custom_objects={"KerasLayer":hub.KerasLayer})
L_M

classes = ("Maybe But Don't get your hopes up You Creep 😂 !","Just Don't Even think about it :)😆")

def predict_this(im):
    if im is not None:
        try:
            img = tf.image.resize(im,size=(224,224))
            img = tf.expand_dims(img,axis=0)  
            pred_prob = L_M.predict(img)
            time.sleep(2)
            return dict(zip(classes,map(float,pred_prob[0])))
        except:
            print("None")


image = gr.inputs.Image(shape=(224,224))
label = gr.outputs.Label()
example = ["p1.jpg","s.jpg","s1.png","a.jpg","b2.png","z.jpg","angel.jpg"]

intf = gr.Interface(fn=predict_this,inputs=image,outputs=label,examples=example).launch()

