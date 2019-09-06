from inference import run_eval
from model import resnet50

truth_class_keys = {'croco': 'n01697457',
                    'tabby_cat': 'n02123045',
                    'wombat': 'n01883070',
                    'bulbul': 'n01560419'}
confidence_thr = 90
img_tabbyCat, classname_tabbyCat, max_tabbyCat, class_key_tabbyCat = run_eval('data/cat.jpg', resnet50())
img_croco, classname_croco, max_croco, class_key_croco = run_eval('data/croco.jpg', resnet50())
img_wombat, classname_wombat, max_wombat, class_key_wombat = run_eval('data/wombat.jpg', resnet50())
img_bulbul, classname_bulbul, max_bulbul, class_key_bulbul = run_eval('data/bulbul.jpg', resnet50())


def test_croco_class_prediction():
    assert class_key_croco == truth_class_keys['croco'], "test failed"
    print('\ntest_croco_class_prediction passed')


def test_tabbyCat_class_prediction():
    assert class_key_tabbyCat == truth_class_keys['tabby_cat'], "test failed"
    print('\ntest_tabbyCat_class_prediction passed')


def test_wombat_class_prediction():
    assert class_key_wombat == truth_class_keys['wombat'], "test failed"
    print('\ntest_wombat_class_prediction passed')


def test_bulbul_class_prediction():
    assert class_key_bulbul == truth_class_keys['bulbul'], "test failed"
    print('\ntest_bulbul_class_prediction passed')


def test_wombat_confidence_threshold():
    assert round(max_wombat.item() * 100, 2) >= confidence_thr, "test failed"
    print('\ntest_wombat_confidence_threshold passed')


def test_bulbul_confidence_threshold():
    assert round(max_bulbul.item() * 100, 2) >= confidence_thr, "test failed"
    print('\ntest_bulbul_confidence_threshold passed')


def test_croco_confidence_threshold():
    assert round(max_croco.item() * 100, 2) >= confidence_thr, "test failed"
    print('\ntest_croco_confidence_threshold passed')
