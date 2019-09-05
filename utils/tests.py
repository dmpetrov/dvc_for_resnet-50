from inference import run_eval
from model import resnet50

truth_class_keys = {'croco': 'n01697457'}
confidence_thr = 65
img, classname, max_, class_key = run_eval('data', resnet50())


# by default croco image is tested
def test_img_predicted():
    assert class_key == truth_class_keys['croco'], "test succeed because class_key=" + str(
        class_key) + " truth_class_key=" + str(truth_class_keys['croco'])


def test_confidence_min_val():
    assert round(max_.item() * 100, 2) >= confidence_thr, "test succeed because predicted_confidence=" + str(
        round(max_.item() * 100, 2)) + " confidence_thr=" + str(confidence_thr)
