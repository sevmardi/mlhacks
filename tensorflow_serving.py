
def serving_input_receiver_fn():
    input_features = tf.placeholder(dtype=tf.string, shape=[None, 1], name='input')
    fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
        features={'input_feature': input_feature})
    return fn

def export_estimator_model(model_fn):
    estimator = tf.estimator.Estimator(model_fn, 'model', params={})
    estimator.export_saved_model('saved_models/', serving_input_receiver_fn)
