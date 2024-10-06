# tests/test_build_roots.py

import unittest
import os
import numpy as np
import logging
from main import compute_roots, save_roots, heat_map

logging.basicConfig(
    filename='test_log.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)


class TestIO(unittest.TestCase):

    def setUp(self):
        """每个测试用例运行之前调用。"""
        logging.info("Setting up test case environment")
        self.degree = 5
        self.tmp_file = "roots_degree_5_test.npy"

    def tearDown(self):
        """每个测试用例运行之后调用。"""
        logging.info("Cleaning up after test case")
        if os.path.exists(self.tmp_file):
            os.remove(self.tmp_file)

    def test_compute_roots(self):
        """测试 compute_roots 函数"""
        logging.info("Starting test: test_compute_roots")
        coeffs = [1, 0, -4]
        roots = compute_roots(coeffs)
        expected_roots = np.array([2, -2])
        self.assertTrue(np.allclose(np.sort(roots), np.sort(expected_roots)),
                        f"Expected {expected_roots}, but got {roots}")
        logging.info(f"Test test_compute_roots passed.")

    def test_save_roots(self):
        """测试 save_roots 函数，确保文件正确保存"""
        logging.info("Starting test: test_save_roots")
        save_roots(self.degree, self.tmp_file)

        self.assertTrue(os.path.exists(self.tmp_file), f"File {self.tmp_file} was not created.")
        roots = np.load(self.tmp_file)
        self.assertGreater(roots.size, 0, "Expected non-empty root data")
        logging.info(f"Test test_save_roots passed.")

    def test_heat_map(self):
        """测试 heat_map 函数"""
        logging.info("Starting test: test_heat_map")
        test_file = "test_roots.npy"
        np.save(test_file, np.array([1 + 1j, -1 - 1j]))

        try:
            img = heat_map(100, test_file)
            self.assertEqual(img.shape, (100, 100), f"Expected 100x100 heatmap, got {img.shape}")
            logging.info("Heatmap test passed.")
        finally:
            if os.path.exists(test_file):
                os.remove(test_file)

    def test_save_roots_degree_5(self):
        """测试生成 degree 5 的根并与 origin_degree_5.npy 进行比较"""
        logging.info("Starting test: test_save_roots_degree_5")
        origin_file = os.path.join(os.path.dirname(__file__), "origin_degree_5.npy")

        # 保存根到临时文件
        save_roots(self.degree, self.tmp_file)

        # 比较生成的根与预期结果
        generated_roots = np.load(self.tmp_file)
        origin_roots = np.load(origin_file)

        generated_roots_sorted = sorted(generated_roots, key=lambda x: (x.real, x.imag))
        origin_roots_sorted = sorted(origin_roots, key=lambda x: (x.real, x.imag))

        self.assertEqual(len(generated_roots_sorted), len(origin_roots_sorted), "Number of roots mismatch.")
        for gen_root, orig_root in zip(generated_roots_sorted, origin_roots_sorted):
            self.assertAlmostEqual(gen_root.real, orig_root.real, places=6, msg="Real parts mismatch.")
            self.assertAlmostEqual(gen_root.imag, orig_root.imag, places=6, msg="Imaginary parts mismatch.")

        logging.info(f"Test test_save_roots_degree_5 passed.")


if __name__ == '__main__':
    unittest.main()
