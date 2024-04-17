import tensorflow as tf
import os
import shap
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import datetime


@dataclass(frozen=True)
class DNNExceptionData:
    data: str


class DNNException(Exception):
    def __init__(self, exception_details: DNNExceptionData):
        self.details = exception_details

    def to_string(self):
        return self.details.data

# Create entry point to Instantiate XAI class with default DNN structure containing:
#     Dense, Input, Hidden, Output layers and Dropout layers
class XAI:
    def __init__(self, model_name, x, y, structure={'core': [{'id': 'input', 'type': 'Dense', 'neurons': 3},
                                                    {'id': 'hidden', 'type': 'Dense',
                                                        'neurons': 10},
                                                    {'id': 'output', 'type': 'Dense', 'neurons': 1}],
                                                    'hp': {'dropout_rate': 0.1, 'learning_rate': 0.01}}):
        self.structure = structure
        self.feature_names = x.columns.values
        self.model_name = model_name
        self.x = x
        self.y = y

#       Train-Test split 80:20
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,
                                                                                y,
                                                                                test_size=0.20,
                                                                                random_state=0)
        scaler = StandardScaler()
#       Important to fit on train only and transform on test to avoid data leakage
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)                                                                        
        self.num_of_features = self.x.shape[1]

        for ix, layer in enumerate(structure['core']):
            if layer['id'] == 'input':
                structure['core'][ix]['neurons'] = self.num_of_features

        self.log_dir = "./logs/" + model_name + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#       Instantiate the DNN model
        self.dnn = DeepNeuralNetwork(structure=self.structure)

#   Function which implements the hyperameter tuning and training with the best HPs
    def fit(self, max_epochs=150, produce_tensorboard=True, explainable=True):
        _callbacks = []
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=15, start_from_epoch=70)
        _callbacks.append(stop_early)

        if produce_tensorboard:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir, histogram_freq=1)
            _callbacks.append(tensorboard_callback)

        hp_tuner = kt.Hyperband(self.dnn.model_builder, objective='val_accuracy', max_epochs=max_epochs, factor=5,
                                directory=f'./hp_{self.model_name}/' + self.model_name, project_name='kt_hb_' + self.model_name)

        hp_tuner.search(self.x_train, self.y_train, epochs=max_epochs, validation_split=0.2, callbacks=[
                        tf.keras.callbacks.TensorBoard('./hp/tb_logs/')])
        best_hpm = hp_tuner.get_best_hyperparameters(num_trials=1)[0]

#       Model with the best hp fit
        self.dnn.model = hp_tuner.hypermodel.build(best_hpm)
# Fit the model to the data
        self.dnn.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),
                           epochs=max_epochs, callbacks=_callbacks)
# get explanations        
        if explainable:
            self.get_explanations(
                self.dnn.model, self.x_train, self.x_test[:100], visualisations=True)
            
        return self.dnn.model, self.x_train, self.x_test
    def get_explanations(self, model, background_data, input_data, visualisations=True):

        self.explainer = shap.DeepExplainer(
            model, background_data
        )
        self.shap_values = self.explainer.shap_values(input_data)
        if visualisations == True:
            self.get_visualisations(self.shap_values, self.explainer)
# get visualisations
    def get_visualisations(self, shap_values, explainer,
                           decision_plot=True,
                           force_plot=True,
                           waterfall_plot=True,
                           summary_plot=True):
        xai_logs_path = './xai_logs/'
        if summary_plot:
            shap.summary_plot(shap_values[0],
                              feature_names=self.feature_names, plot_type='bar', show=False)
            plt.savefig(
                xai_logs_path + self.model_name + '_summary_plot.png', bbox_inches='tight', dpi=600)
            plt.close()
        if force_plot:
            shap.force_plot(explainer.expected_value[0].numpy(),
                            shap_values[0][0], features=self.feature_names, matplotlib=True, show=False)
            plt.savefig(
                xai_logs_path + self.model_name + '_force_plot.png', bbox_inches='tight', dpi=600)
            plt.close()

        if waterfall_plot:
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0].numpy(),
                                                   shap_values[0][0],
                                                #    features=self.x_test.iloc[0, :],
                                                   feature_names=self.feature_names, show=False)
            plt.savefig(
                xai_logs_path + self.model_name + '_waterfall_plot.png', bbox_inches='tight', dpi=600)
            plt.close()

        if decision_plot:
            shap.decision_plot(explainer.expected_value[0].numpy(),
                               shap_values[0][0],
                            #    features=self.x_test.iloc[0, :],
                               feature_names=self.feature_names.tolist(), show=False)
            plt.savefig(
                xai_logs_path + self.model_name + '_decision_plot.png', bbox_inches='tight', dpi=600)
            plt.close()

# DNN model
class DeepNeuralNetwork(tf.keras.Model):
    def __init__(self, structure):
        super(DeepNeuralNetwork, self).__init__(name='DNN')
        self.structure = structure

    def model_builder(self, hp):
        _model = tf.keras.Sequential()

#       set-up hyperparameter space
        hp_neurons = hp.Int('units', min_value=10, max_value=40, step=10)
        hp_dropout_rate = hp.Choice('dropout_rate', values=[ 0.1, 0.2, 0.3])
        hp_learning_rate = hp.Choice(
            'learning_rate', values=[1e-2, 1e-3, 1e-4]) 
        activation = hp.Choice('activation', values=['sigmoid', 'elu', 'relu'])

        for ix, layer in enumerate(self.structure['core']):
            if layer['id'] == 'input':
                _model.add(tf.keras.layers.Flatten(
                    input_shape=(layer['neurons'], )))
                exec(
                    f"_model.add(tf.keras.layers.{layer['type']}({layer['neurons']}, trainable=True))")
                _model.add(tf.keras.layers.Dropout(hp_dropout_rate))
            elif layer['id'] == 'hidden':
                exec(
                    f"_model.add(tf.keras.layers.{layer['type']}(hp_neurons, activation=activation, trainable=True))")
                _model.add(tf.keras.layers.Dropout(hp_dropout_rate))
            else:
                exec(
                    f"_model.add(tf.keras.layers.{layer['type']}({layer['neurons']}, trainable=True))")
# Model Compile - loss function ==> Binary Crossentropy and we track accuracy                    
        _model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate,
                                                                 epsilon=1e-07),
                       loss='binary_crossentropy',  
                       metrics=['accuracy'])
# return the model to be used in the XAI function
        return _model
