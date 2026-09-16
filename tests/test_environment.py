"""
Environment verification test.
Checks Python version and core library imports, prints PyTorch and CUDA info,
and ensures CPU/GPU execution is properly handled.
"""
import sys
import unittest


class TestEnvironment(unittest.TestCase):
    def test_python_version(self):
        print(f"\n[INFO] Python Version: {sys.version.split()[0]}")
        self.assertGreaterEqual(sys.version_info, (3, 9))

    def test_numpy_import(self):
        import numpy as np
        print(f"[INFO] NumPy Version: {np.__version__}")
        self.assertIsNotNone(np.__version__)

    def test_pandas_import(self):
        import pandas as pd
        print(f"[INFO] Pandas Version: {pd.__version__}")
        self.assertIsNotNone(pd.__version__)

    def test_scipy_import(self):
        import scipy
        print(f"[INFO] SciPy Version: {scipy.__version__}")
        self.assertIsNotNone(scipy.__version__)

    def test_sklearn_import(self):
        import sklearn
        print(f"[INFO] scikit-learn Version: {sklearn.__version__}")
        self.assertIsNotNone(sklearn.__version__)

    def test_xgboost_import(self):
        import xgboost as xgb
        print(f"[INFO] XGBoost Version: {xgb.__version__}")
        self.assertIsNotNone(xgb.__version__)

    def test_torch_and_cuda(self):
        import torch
        print(f"[INFO] PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"[INFO] torch.cuda.is_available(): {cuda_available}")

        if cuda_available:
            print(f"[INFO] CUDA Version: {torch.version.cuda}")
            print(f"[INFO] GPU Device Count: {torch.cuda.device_count()}")
            print(f"[INFO] GPU Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] CUDA is not available. System is operating in CPU mode.")
        self.assertIsNotNone(torch.__version__)

    def test_transformers_import(self):
        import transformers
        print(f"[INFO] Transformers Version: {transformers.__version__}")
        self.assertIsNotNone(transformers.__version__)

    def test_sentence_transformers_import(self):
        import sentence_transformers
        print(f"[INFO] Sentence-Transformers Version: {sentence_transformers.__version__}")
        self.assertIsNotNone(sentence_transformers.__version__)

    def test_datasets_import(self):
        import datasets
        print(f"[INFO] Datasets Version: {datasets.__version__}")
        self.assertIsNotNone(datasets.__version__)

    def test_faiss_import(self):
        import faiss
        print(f"[INFO] FAISS Version: {faiss.__version__}")
        self.assertIsNotNone(faiss.__version__)

    def test_streamlit_import(self):
        import streamlit as st
        print(f"[INFO] Streamlit Version: {st.__version__}")
        self.assertIsNotNone(st.__version__)


def main():
    print("=" * 60)
    print("RUNNING ENVIRONMENT VERIFICATION")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnvironment)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("=" * 60)
    if result.wasSuccessful():
        print("All environment checks passed successfully!")
    else:
        print("Environment verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
