"""Check installed launchers, image loading and FEAT predictions."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import shutil
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
from PySide6 import QtWidgets

from bold_viewer.viewer import BoldViewer, read_fsl_matrix


@pytest.fixture(scope="session")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_installed_commands():
    launcher = shutil.which("call_viewer")
    assert launcher is not None
    for command in ([launcher, "--help"], [sys.executable, "-m", "bold_viewer", "--help"]):
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert "BOLD / FEAT viewer" in result.stdout


def test_image_and_overlay(app, tmp_path):
    base = tmp_path / "image with spaces.nii.gz"
    overlay = tmp_path / "overlay.nii.gz"
    data = np.arange(120, dtype=np.float32).reshape(3, 4, 5, 2)
    nib.save(nib.Nifti1Image(data, np.eye(4)), base)
    nib.save(nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.float32), np.eye(4)), overlay)
    viewer = BoldViewer()
    try:
        viewer.load_base(base)
        viewer.load_overlay(overlay)
        viewer.show()
        app.processEvents()
        np.testing.assert_array_equal(viewer.base_layer.data, data[::-1, ::-1, ::-1])
        assert viewer.base_layer.nvolumes == 2
        assert len(viewer.maps) == 1
        assert viewer._collect_series()[0]["bold"][1].size == 2
    finally:
        viewer.close()


def test_feat_prediction(app, tmp_path):
    folder = tmp_path / "example.feat"
    (folder / "stats").mkdir(parents=True)
    design = np.array([[0., 1.], [1., 1.], [2., 1.]])
    (folder / "design.mat").write_text("/NumWaves 2\n/NumPoints 3\n/Matrix\n0 1\n1 1\n2 1\n")
    (folder / "design.con").write_text("/NumWaves 2\n/NumContrasts 1\n/Matrix\n1 0\n")
    def save(path, values):
        nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32), np.eye(4)), path)
    save(folder / "filtered_func_data.nii.gz", np.broadcast_to([10., 12., 14.], (2, 2, 2, 3)).copy())
    save(folder / "stats" / "pe1.nii.gz", np.full((2, 2, 2), 2.))
    save(folder / "stats" / "pe2.nii.gz", np.zeros((2, 2, 2)))
    viewer = BoldViewer()
    try:
        viewer.load_base(folder)
        np.testing.assert_array_equal(viewer.design, design)
        assert viewer._design_ready()
        for i in range(viewer.trace_table.topLevelItemCount()):
            item = viewer.trace_table.topLevelItem(i)
            from PySide6 import QtCore
            if item.data(0, QtCore.Qt.UserRole) == "prediction":
                item.setCheckState(0, QtCore.Qt.Checked)
        series, _ = viewer._collect_series()
        expected = design[:, 0] * 2
        if not viewer.demean.isChecked():
            expected += 12
        np.testing.assert_allclose(series["prediction"][1], expected)
    finally:
        viewer.close()


def test_single_column_design(tmp_path):
    path = tmp_path / "design.mat"
    path.write_text("/NumWaves 1\n/NumPoints 2\n/Matrix\n1\n2\n")
    assert read_fsl_matrix(path).shape == (2, 1)
