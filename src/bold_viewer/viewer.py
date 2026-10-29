#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""PySide6 BOLD/NIfTI and FEAT viewer.

Install: python -m pip install pyside6 pyqtgraph nibabel numpy
Run:     call_viewer [image.nii.gz | analysis.feat] [-o map.nii.gz ...]

Image, Maps and Plots tabs are scrollable; drag the central splitters to resize.
Map order is bottom to top. Each map has its own visibility, opacity, threshold,
color range and palette. Overlays must already share the base grid AND affine;
this application does not perform registration/resampling.

Open FEAT discovers design.mat, design.con, stats/pe*, stats/cope* and res4d.
EV and contrast names are read from design.fsf and design.con; the Design tab
shows real-column PE names, signed contrast definitions and a map loader.
Select individual EVs, PE contributions, COPE partial fits, residuals, predictions
and coefficient values in Plots. Double-click any trace to choose its color.
The selected-PE prediction sums checked 'PE: fitted contribution' rows. The
PE/COPE values plot shows scalar coefficients, independently of time courses.

Model calculations:
- Full prediction: X @ beta, evaluated on the fitted design (in-sample).
- PE contribution: X[:, i] * beta[i].
- COPE partial fit: sum(X[:, i] * beta[i] * abs(c_normalized[i])), following
  FSLeyes display convention, c_normalized = c * sqrt(count_nonzero(c))/norm(c).
  This is a partial model display, not the scalar contrast estimate c @ beta.
- Residual-implied fit: filtered_func_data - stats/res4d.
- First-level design predictions add the voxel data mean unless Demean is on.
  Demean subtracts the data mean from BOLD/residual-implied fit; residuals and
  raw EV columns retain their own units. PE/COPE scalar values are unchanged.

Design curves require filtered_func_data, matching design rows and relevant PE
maps. Voxelwise EV designs are detected via design.fsf and disabled because a
single design.mat is insufficient. Curves are not prewhitened; FEAT report
plots may differ. All coordinates are zero based in the internal LPI grid;
status bar world coordinates are RAS mm. Both .nii and .nii.gz are supported.
CSV exports checked curves and scalar values (in row zero), using display units.
"""

from __future__ import annotations

import os
os.environ.setdefault("QT_API", "pyside6")

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict
import csv
import io
import re

import numpy as np
import nibabel as nib
import nibabel.orientations as nio
from PySide6 import QtCore, QtWidgets, QtGui
import pyqtgraph as pg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

Array = np.ndarray
Voxel = Tuple[int, int, int]
CANONICAL_AXCODES = ("L", "P", "I")


@dataclass
class DisplayRange:
    vmin: float
    vmax: float

    def valid(self) -> bool:
        return np.isfinite(self.vmin) and np.isfinite(self.vmax) and self.vmax > self.vmin


@dataclass
class NiftiLayer:
    path: Path
    img: nib.spatialimages.SpatialImage
    data: Array
    affine: Array
    name: str
    original_axcodes: Tuple[str, str, str]

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def is_4d(self) -> bool:
        return self.data.ndim == 4

    @property
    def nvolumes(self) -> int:
        return int(self.data.shape[3]) if self.is_4d else 1

    def volume(self, t: int = 0) -> Array:
        if self.is_4d:
            return self.data[..., int(np.clip(t, 0, self.nvolumes - 1))]
        return self.data


@dataclass
class ViewerState:
    voxel: Voxel = (0, 0, 0)
    volume: int = 0
    overlay_visible: bool = True
    overlay_alpha: float = 0.45
    overlay_pos_min: float = 2.3
    overlay_pos_max: float = 8.0
    overlay_neg_min: float = -8.0
    overlay_neg_max: float = -2.3
    base_range: Optional[DisplayRange] = None
    overlay_range: Optional[DisplayRange] = None
    auto_ts_scale: bool = True


class RangeWidget(QtWidgets.QGroupBox):
    """Small min/max range control with reset button."""

    range_changed = QtCore.Signal()
    reset_requested = QtCore.Signal()

    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(title, parent)
        layout = QtWidgets.QFormLayout(self)
        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.min_spin, self.max_spin):
            spin.setDecimals(4)
            spin.setRange(-1e12, 1e12)
            spin.setSingleStep(0.1)
        self.reset_button = QtWidgets.QPushButton("Reset range")
        layout.addRow("Min", self.min_spin)
        layout.addRow("Max", self.max_spin)
        layout.addRow(self.reset_button)
        self.min_spin.valueChanged.connect(self.range_changed)
        self.max_spin.valueChanged.connect(self.range_changed)
        self.reset_button.clicked.connect(self.reset_requested)

    def set_range(self, vmin: float, vmax: float) -> None:
        self.blockSignals(True)
        self.min_spin.setValue(float(vmin))
        self.max_spin.setValue(float(vmax))
        self.blockSignals(False)

    def range(self) -> DisplayRange:
        return DisplayRange(float(self.min_spin.value()), float(self.max_spin.value()))


class SliceView(pg.GraphicsLayoutWidget):
    """One anatomical slice panel with draggable crosshair."""

    voxel_changed = QtCore.Signal(tuple)
    view_reset_requested = QtCore.Signal()

    def __init__(self, orientation: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.orientation = orientation
        self.view = self.addViewBox(row=0, col=0, lockAspect=True, enableMenu=False)
        # Allow zoom/pan. Crosshair dragging still works via InfiniteLine, and
        # click/drag in the image updates the selected voxel.
        self.view.setMouseEnabled(x=True, y=True)
        self.base_item = pg.ImageItem(axisOrder="row-major")
        self.overlay_item = pg.ImageItem(axisOrder="row-major")
        self.view.addItem(self.base_item)
        self.view.addItem(self.overlay_item)

        self.vline = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen((255, 235, 0), width=1))
        self.hline = pg.InfiniteLine(angle=0, movable=True, pen=pg.mkPen((255, 235, 0), width=1))
        self.view.addItem(self.vline)
        self.view.addItem(self.hline)
        self.vline.sigDragged.connect(self._line_dragged)
        self.hline.sigDragged.connect(self._line_dragged)

        self.label = pg.TextItem(text=orientation, color=(230, 230, 230), anchor=(0, 1))
        self.view.addItem(self.label)
        self._shape: Optional[Tuple[int, int]] = None
        self._range_initialised = False
        self._updating_crosshair = False

    def set_images(
        self,
        base: Array,
        base_range: DisplayRange,
        overlay: Optional[Array],
        overlay_range: DisplayRange,
        overlay_alpha: float,
        overlay_pos_min: float,
        overlay_pos_max: float,
        overlay_neg_min: float,
        overlay_neg_max: float,
    ) -> None:
        base2d = np.nan_to_num(np.asarray(base, dtype=np.float32))
        self._shape = base2d.shape
        self.base_item.setImage(base2d, levels=(base_range.vmin, base_range.vmax), autoLevels=False)

        if overlay is None:
            self.overlay_item.clear()
        else:
            ov = np.asarray(overlay, dtype=np.float32)
            rgba = self._overlay_to_rgba(
                ov,
                overlay_range,
                overlay_alpha,
                overlay_pos_min,
                overlay_pos_max,
                overlay_neg_min,
                overlay_neg_max,
            )
            self.overlay_item.setImage(rgba, autoLevels=False)

        if not self._range_initialised:
            self.reset_view_range()
            self._range_initialised = True

    def reset_view_range(self) -> None:
        if self._shape is None:
            return
        rows, cols = self._shape
        self.view.setRange(xRange=(0, cols), yRange=(0, rows), padding=0.02)
        self._range_initialised = True

    def set_crosshair(self, xy: Tuple[int, int]) -> None:
        self._updating_crosshair = True
        x, y = xy
        self.vline.setPos(x + 0.5)
        self.hline.setPos(y + 0.5)
        self.label.setPos(6, 14)
        self._updating_crosshair = False

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = self.view.mapSceneToView(event.position())
        self._emit_position(pos.x(), pos.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.buttons() & QtCore.Qt.LeftButton:
            pos = self.view.mapSceneToView(event.position())
            self._emit_position(pos.x(), pos.y())
        super().mouseMoveEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        # Do not zoom panes accidentally; wheel belongs to the parent scroll/volume controls later.
        event.ignore()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        reset = menu.addAction("Reset view")
        chosen = menu.exec(event.globalPos())
        if chosen == reset:
            self.view_reset_requested.emit()

    def _line_dragged(self) -> None:
        if self._updating_crosshair:
            return
        self._emit_position(float(self.vline.value()), float(self.hline.value()))

    def _emit_position(self, x: float, y: float) -> None:
        if self._shape is None:
            return
        rows, cols = self._shape
        xi = int(np.clip(np.floor(x), 0, cols - 1))
        yi = int(np.clip(np.floor(y), 0, rows - 1))
        self.voxel_changed.emit((self.orientation, xi, yi))

    @staticmethod
    def _overlay_to_rgba(
        data: Array,
        value_range: DisplayRange,
        alpha: float,
        pos_min: float,
        pos_max: float,
        neg_min: float,
        neg_max: float,
    ) -> Array:
        rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
        if not value_range.valid():
            return rgba

        pos_lo, pos_hi = sorted((float(pos_min), float(pos_max)))
        neg_lo, neg_hi = sorted((float(neg_min), float(neg_max)))
        pos = (data >= pos_lo) & (data <= pos_hi)
        neg = (data >= neg_lo) & (data <= neg_hi)
        if not np.any(pos | neg):
            return rgba

        pden = max(pos_hi - pos_lo, 1e-6)
        nden = max(neg_hi - neg_lo, 1e-6)
        pscaled = np.clip((data - pos_lo) / pden, 0, 1)
        nscaled = np.clip((neg_hi - data) / nden, 0, 1)

        rgba[..., 0][pos] = 255
        rgba[..., 1][pos] = (80 + 175 * pscaled[pos]).astype(np.uint8)
        rgba[..., 2][pos] = 0
        rgba[..., 0][neg] = 0
        rgba[..., 1][neg] = (80 + 120 * nscaled[neg]).astype(np.uint8)
        rgba[..., 2][neg] = 255
        rgba[..., 3][pos | neg] = int(np.clip(alpha, 0, 1) * 255)
        return rgba


@dataclass
class MapStyle:
    layer: NiftiLayer
    visible: bool = True
    alpha: float = 0.45
    low: float = -1.0
    high: float = 1.0
    threshold: float = 0.0
    palette: str = "Signed"
    positive: str = "#ffb000"
    negative: str = "#25aaff"


def read_fsl_matrix(path: Path) -> Array:
    """Read a VEST design/contrast matrix without collapsing singleton axes."""
    content = path.read_text()
    if '/Matrix' not in content:
        raise ValueError(f'{path.name}: missing /Matrix')
    header, body = content.split('/Matrix', 1)
    matrix = np.loadtxt(io.StringIO(body), ndmin=2)
    waves = re.search(r'/NumWaves\s+(\d+)', header)
    rows = re.search(r'/(?:NumPoints|NumContrasts)\s+(\d+)', header)
    if waves and matrix.shape[1] != int(waves[1]):
        raise ValueError(f'{path.name}: incorrect number of columns')
    if rows and matrix.shape[0] != int(rows[1]):
        raise ValueError(f'{path.name}: incorrect number of rows')
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f'{path.name}: nonfinite entries')
    return matrix


def _feat_text(value):
    """Decode literal FEAT/Tcl names without evaluating the settings file."""
    value = value.strip()
    if value.startswith('"'):
        match = re.match(r'"((?:\\.|[^"\\])*)"', value)
        if match:
            return re.sub(r'\\(["\\])', r'\1', match[1])
    if value.startswith('{') and value.endswith('}'):
        return value[1:-1]
    return value.split(' #', 1)[0].strip()


def read_feat_names(folder, ncolumns):
    """Map real design columns/contrast numbers to FEAT titles.

    Derivatives and basis EVs occupy additional PE columns. Never zip original
    EV titles directly to PE numbers. Unspecified trailing columns stay unnamed.
    """
    settings, evs, contrasts, warnings = {}, {}, {}, []
    fsf = folder / 'design.fsf'
    if fsf.exists():
        for line in fsf.read_text(errors='replace').splitlines():
            match = re.match(r'\s*set\s+fmri\(([^)]+)\)\s+(.+)', line)
            if match:
                settings[match[1]] = _feat_text(match[2])
    def integer(key, default=0):
        try:
            return int(settings.get(key, default))
        except ValueError:
            warnings.append(f'Invalid {key} in design.fsf; check the column mapping.')
            return default
    originals = integer('evs_orig', max([int(k[7:]) for k in settings
                           if re.fullmatch(r'evtitle\d+', k)] + [0]))
    names = []
    for i in range(1, originals+1):
        title = settings.get(f'evtitle{i}', '').strip() or f'Original EV {i} (unnamed)'
        names.append(title)
        if integer('level', 1) == 1:
            if integer(f'convolve{i}') in (4, 5, 6):
                count = integer(f'basisfnum{i}', -1)
                if count < 1:
                    warnings.append(f'EV {i}: missing basis count; EV column names were not assigned.')
                    names = []
                    break
                names.extend(f'{title} [basis {b}]' for b in range(2, count+1))
            if integer(f'deriv_yn{i}') == 1:
                names.append(f'{title} [temporal derivative]')
    if len(names) > ncolumns:
        warnings.append('EV metadata exceeds design columns; EV column names were not assigned.')
        names = []
    for i in range(1, ncolumns+1):
        evs[i] = names[i-1] if i <= len(names) else 'Unnamed design column'
    for key, value in sorted(settings.items(), key=lambda pair: pair[0].startswith('conname_real.')):
        match = re.fullmatch(r'conname_(?:real\.)?(\d+)', key)
        if match and value.strip():
            contrasts[int(match[1])] = value
    con = folder / 'design.con'
    if con.exists():
        for line in con.read_text(errors='replace').split('/Matrix', 1)[0].splitlines():
            match = re.match(r'\s*/ContrastName(\d+)\s+(.+)', line)
            if match and _feat_text(match[2]):
                contrasts[int(match[1])] = _feat_text(match[2])
    return evs, contrasts, settings, warnings


class BoldViewer(QtWidgets.QMainWindow):
    """Orthogonal viewer with independent map styles and FEAT plot selection."""

    COLORS = ['#e6e6e6', '#00c8ff', '#ffb000', '#a8dc55', '#ef74be', '#aa99ff']

    def __init__(self, image_path=None, overlay_path=None):
        super().__init__()
        self.setWindowTitle('BOLD / FEAT Viewer')
        self.resize(1450, 900)
        self.state = ViewerState()
        self.base_layer = None
        self.maps = []
        self.feat_dir = None
        self.design = None
        self.contrasts = None
        self.pe_layers = {}
        self.cope_layers = {}
        self.residual_layer = None
        self._refreshing = False
        self._editing = False
        self._curves = {}
        self._series = {}
        self.ev_names = {}
        self.contrast_names = {}
        self._build_ui()
        if image_path is not None:
            self.safe_load(self.load_base, Path(image_path))
        if overlay_path is not None:
            self.safe_load(self.load_overlay, Path(overlay_path))

    def _button(self, text, callback):
        button = QtWidgets.QPushButton(text)
        button.clicked.connect(lambda checked=False: callback())
        return button

    def _scroll_tab(self, title):
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setSpacing(10)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, title)
        return layout

    def _build_ui(self):
        toolbar = self.addToolBar('Files')
        toolbar.setMovable(False)
        for title, callback in [('Open image', self._open_base_dialog), ('Open FEAT', self._open_feat_dialog),
                                ('Add maps', self._open_overlay_dialog), ('Export curves', self._save_time_series)]:
            action = toolbar.addAction(title)
            action.triggered.connect(lambda checked=False, cb=callback: cb())
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMinimumWidth(295)
        self.tabs.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Ignored)
        image_layout = self._scroll_tab('Image')
        maps_layout = self._scroll_tab('Maps')
        plot_layout = self._scroll_tab('Plots')
        design_layout = self._scroll_tab('Design')
        self.info_label = QtWidgets.QLabel('Open a NIfTI image or FEAT directory.')
        self.info_label.setWordWrap(True)
        self.info_label.setTextFormat(QtCore.Qt.PlainText)
        image_layout.addWidget(self.info_label)
        group = QtWidgets.QGroupBox('Voxel / volume (zero based)')
        form = QtWidgets.QFormLayout(group)
        self.x_spin, self.y_spin, self.z_spin, self.t_spin = [QtWidgets.QSpinBox() for _ in range(4)]
        for label, spin in zip(('LPI X', 'LPI Y', 'LPI Z', 'Volume'), (self.x_spin, self.y_spin, self.z_spin, self.t_spin)):
            spin.setKeyboardTracking(False)
            form.addRow(label, spin)
        for spin in (self.x_spin, self.y_spin, self.z_spin):
            spin.valueChanged.connect(self._spin_voxel_changed)
        self.t_spin.valueChanged.connect(self._set_volume)
        self.t_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.t_slider.valueChanged.connect(self._set_volume)
        form.addRow(self.t_slider)
        form.addRow(self._button('Center voxel', self.center_voxel))
        form.addRow(self._button('Reset slice views', self.reset_views))
        image_layout.addWidget(group)
        self.base_range = RangeWidget('Base intensity')
        self.base_range.range_changed.connect(self._range_controls_changed)
        self.base_range.reset_requested.connect(self.reset_base_range)
        image_layout.addWidget(self.base_range)
        image_layout.addStretch()

        maps_layout.addWidget(QtWidgets.QLabel('Checked maps are drawn bottom to top.'))
        self.map_list = QtWidgets.QListWidget()
        self.map_list.setMinimumHeight(130)
        self.map_list.setMaximumHeight(220)
        self.map_list.currentRowChanged.connect(self._select_map)
        self.map_list.itemChanged.connect(self._map_visibility)
        maps_layout.addWidget(self.map_list)
        row = QtWidgets.QHBoxLayout()
        for text, callback in [('Add', self._open_overlay_dialog), ('Remove', self._remove_map),
                               ('↑', lambda: self._move_map(-1)), ('↓', lambda: self._move_map(1))]:
            row.addWidget(self._button(text, callback))
        maps_layout.addLayout(row)
        self.map_editor = QtWidgets.QGroupBox('Selected map')
        form = QtWidgets.QFormLayout(self.map_editor)
        self.map_alpha = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.map_alpha.setRange(0, 100)
        self.map_palette = QtWidgets.QComboBox()
        self.map_palette.addItems(['Signed', 'viridis', 'magma', 'grey'])
        self.map_low, self.map_high, self.map_threshold = [QtWidgets.QDoubleSpinBox() for _ in range(3)]
        for spin in (self.map_low, self.map_high, self.map_threshold):
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(5)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._map_style_changed)
        self.map_threshold.setMinimum(0)
        self.pos_button = self._button('Positive color', lambda: self._map_color('positive'))
        self.neg_button = self._button('Negative color', lambda: self._map_color('negative'))
        form.addRow('Opacity', self.map_alpha)
        form.addRow('Color map', self.map_palette)
        form.addRow('Display min', self.map_low)
        form.addRow('Display max', self.map_high)
        form.addRow('Hide |value| below', self.map_threshold)
        form.addRow(self.pos_button)
        form.addRow(self.neg_button)
        form.addRow(self._button('Reset map range', self._reset_map_range))
        self.map_alpha.valueChanged.connect(self._map_style_changed)
        self.map_palette.currentTextChanged.connect(self._map_style_changed)
        maps_layout.addWidget(self.map_editor)
        note = QtWidgets.QLabel('Maps must share the base voxel grid and affine. Register/resample other spaces before loading. Zero and nonfinite values are transparent.')
        note.setWordWrap(True)
        maps_layout.addWidget(note)
        maps_layout.addStretch()
        self.map_editor.setEnabled(False)

        self.auto_ts_scale = QtWidgets.QCheckBox('Auto-scale plots')
        self.auto_ts_scale.setChecked(True)
        self.auto_ts_scale.toggled.connect(self._update_time_series)
        self.demean = QtWidgets.QCheckBox('Demean signal and fitted curves')
        self.demean.setChecked(True)
        self.demean.toggled.connect(self._update_time_series)
        plot_layout.addWidget(self.auto_ts_scale)
        plot_layout.addWidget(self.demean)
        self.show_legend = QtWidgets.QCheckBox('Show plot legend')
        self.show_legend.toggled.connect(lambda visible: self.ts_plot.plotItem.legend.setVisible(visible))
        plot_layout.addWidget(self.show_legend)
        self.trace_table = QtWidgets.QTreeWidget()
        self.trace_table.setHeaderLabels(['Time course / coefficient', 'Color'])
        self.trace_table.setRootIsDecorated(False)
        self.trace_table.setMinimumHeight(240)
        self.trace_table.setColumnWidth(0, 205)
        self.trace_table.itemChanged.connect(self._trace_changed)
        self.trace_table.itemDoubleClicked.connect(self._trace_color)
        plot_layout.addWidget(self.trace_table)
        plot_layout.addWidget(self._button('Reset plot ranges', lambda: self._update_time_series(True)))
        plot_layout.addWidget(self._button('Export displayed curves CSV', self._save_time_series))
        self.model_info = QtWidgets.QLabel('No FEAT model loaded.')
        self.design_filter = QtWidgets.QLineEdit()
        self.design_filter.setPlaceholderText('Find EV, contrast name or number…')
        self.design_filter.textChanged.connect(self._filter_design)
        design_layout.addWidget(self.design_filter)
        design_layout.addWidget(QtWidgets.QLabel('PE number → real design column'))
        self.ev_table = QtWidgets.QTreeWidget()
        self.ev_table.setHeaderLabels(['PE / EV', 'Name'])
        self.ev_table.setRootIsDecorated(False)
        self.ev_table.setMinimumHeight(160)
        design_layout.addWidget(self.ev_table)
        design_layout.addWidget(QtWidgets.QLabel('Contrast number → name and signed weights'))
        self.contrast_table = QtWidgets.QTreeWidget()
        self.contrast_table.setHeaderLabels(['Contrast', 'Name', 'Definition (Σ weight × PE)'])
        self.contrast_table.setRootIsDecorated(False)
        self.contrast_table.setMinimumHeight(180)
        self.contrast_table.itemDoubleClicked.connect(lambda item, column: self._load_contrast_map())
        design_layout.addWidget(self.contrast_table)
        self.contrast_detail = QtWidgets.QLabel('Select a contrast to inspect its definition.')
        self.contrast_detail.setTextFormat(QtCore.Qt.PlainText)
        self.contrast_detail.setWordWrap(True)
        self.contrast_detail.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.contrast_table.currentItemChanged.connect(self._show_contrast_detail)
        design_layout.addWidget(self.contrast_detail)
        self.contrast_map_kind = QtWidgets.QComboBox()
        self.contrast_map_kind.addItems(['zstat', 'tstat', 'cope', 'varcope'])
        design_layout.addWidget(self.contrast_map_kind)
        design_layout.addWidget(self._button('Overlay selected contrast map', self._load_contrast_map))
        hint = QtWidgets.QLabel('Select a contrast and a map type to overlay it. Numbers match stats/copeN, tstatN and zstatN. Hover over a definition for full names and weights.')
        hint.setWordWrap(True)
        design_layout.addWidget(hint)
        design_layout.addStretch()
        self.model_info.setWordWrap(True)
        plot_layout.addWidget(self.model_info)
        plot_layout.addStretch()

        self.axial, self.coronal, self.sagittal = [SliceView(o) for o in ('axial', 'coronal', 'sagittal')]
        self.slice_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        for view in (self.axial, self.coronal, self.sagittal):
            self.slice_splitter.addWidget(view)
            view.voxel_changed.connect(self._slice_position_changed)
            view.view_reset_requested.connect(self.reset_views)
        self.ts_plot = pg.PlotWidget(title='Voxel time courses')
        self.ts_plot.setLabel('bottom', 'Volume (zero based)')
        self.ts_plot.setLabel('left', 'Signal / model units')
        self.ts_plot.addLegend(offset=(10, 10), colCount=2)
        self.ts_plot.plotItem.legend.setVisible(False)
        self.ts_plot.showGrid(x=True, y=True, alpha=0.15)
        self.t_marker = pg.InfiniteLine(angle=90, pen=pg.mkPen('#e8d54a'))
        self.ts_plot.addItem(self.t_marker)
        self.coef_plot = pg.PlotWidget(title='Selected PE / COPE values at voxel')
        self.coef_plot.setLabel('left', 'Estimate (native units)')
        self.plot_tabs = QtWidgets.QTabWidget()
        self.plot_tabs.addTab(self.ts_plot, 'Time courses')
        self.plot_tabs.addTab(self.coef_plot, 'PE / COPE values')
        main = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        main.addWidget(self.slice_splitter)
        main.addWidget(self.plot_tabs)
        main.setSizes([530, 310])
        self.layout_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.layout_splitter.addWidget(self.tabs)
        self.layout_splitter.addWidget(main)
        self.layout_splitter.setStretchFactor(1, 1)
        self.layout_splitter.setSizes([345, 1100])
        self.setCentralWidget(self.layout_splitter)
        self.status = QtWidgets.QLabel('Ready')
        self.statusBar().addWidget(self.status, 1)
        self._populate_traces()

    def safe_load(self, callback, path):
        try:
            callback(path)
        except Exception as exc:
            logger.exception('Loading failed')
            QtWidgets.QMessageBox.warning(self, 'Unable to load', str(exc))

    def _open_base_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image', '', 'NIfTI (*.nii *.nii.gz)')
        if path:
            self.safe_load(self.load_base, Path(path))

    def _open_feat_dialog(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open FEAT directory')
        if path:
            self.safe_load(self.load_base, Path(path))

    def _open_overlay_dialog(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, 'Add overlay maps', '', 'NIfTI (*.nii *.nii.gz)')
        for path in paths:
            self.safe_load(self.load_overlay, Path(path))

    @staticmethod
    def _find_image(folder, stem):
        return next((folder / (stem + suffix) for suffix in ('.nii.gz', '.nii')
                     if (folder / (stem + suffix)).exists()), None)

    def load_base(self, path):
        path = Path(path).expanduser().resolve()
        feat = path if path.is_dir() else (path.parent if path.parent.suffix == '.feat'
            and path.name in tuple(stem + suffix for stem in ('filtered_func_data', 'example_func', 'mean_func')
                                  for suffix in ('.nii', '.nii.gz')) else None)
        if path.is_dir():
            path = next((p for stem in ('filtered_func_data', 'example_func', 'mean_func')
                         if (p := self._find_image(path, stem)) is not None), None)
            if path is None:
                raise ValueError('No filtered_func_data, example_func or mean_func image found.')
        layer = self._load_nifti_lpi(path, path.name)
        self.base_layer = layer
        self.feat_dir = feat
        self.state.volume = 0
        self.maps.clear()
        self.map_list.clear()
        self._select_map(-1)
        self.center_voxel(False)
        self._configure_spins()
        self.t_slider.setRange(0, layer.nvolumes - 1)
        self.t_spin.setRange(0, layer.nvolumes - 1)
        for widget in (self.t_slider, self.t_spin):
            widget.blockSignals(True)
            widget.setValue(0)
            widget.setEnabled(layer.nvolumes > 1)
            widget.blockSignals(False)
        self.reset_base_range(False)
        self._load_feat()
        self.info_label.setText(f'{path.name}\nLPI shape: {layer.shape}\nOriginal axes: {"".join(layer.original_axcodes)}\nDisplay axes: LPI')
        self.info_label.setToolTip(str(path))
        for view in (self.axial, self.coronal, self.sagittal):
            view._range_initialised = False
        self.refresh()
        if feat:
            for stem in ('zstat1', 'tstat1', 'cope1'):
                overlay = self._find_image(feat / 'stats', stem)
                if overlay:
                    self.safe_load(self.load_overlay, overlay)
                    break

    @staticmethod
    def _same_grid(a, b):
        return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, rtol=1e-5, atol=1e-4)

    def load_overlay(self, path):
        if self.base_layer is None:
            raise ValueError('Load a base image first.')
        layer = self._load_nifti_lpi(path, Path(path).name)
        if not self._same_grid(self.base_layer, layer):
            raise ValueError('Map grid/affine differs from the base image. Register/resample it to the base grid first.')
        style = MapStyle(layer)
        self._default_map_range(style)
        if 'stat' in layer.path.name:
            style.threshold = 2.3
        self.maps.append(style)
        self._editing = True
        item = QtWidgets.QListWidgetItem(self._map_title(layer.path))
        item.setToolTip(str(layer.path))
        item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.Checked)
        self.map_list.addItem(item)
        self._editing = False
        self.map_list.setCurrentRow(len(self.maps) - 1)
        self.refresh()

    @staticmethod
    def _default_map_range(style):
        volume = style.layer.volume(0)
        finite = volume[np.isfinite(volume) & (volume != 0)]
        extent = max(float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0, 1e-6)
        style.low, style.high = -extent, extent

    def _select_map(self, index):
        self.map_editor.setEnabled(0 <= index < len(self.maps))
        if not 0 <= index < len(self.maps):
            return
        self._editing = True
        style = self.maps[index]
        self.map_alpha.setValue(round(100 * style.alpha))
        self.map_palette.setCurrentText(style.palette)
        self.map_low.setValue(style.low)
        self.map_high.setValue(style.high)
        self.map_threshold.setValue(style.threshold)
        self.pos_button.setStyleSheet(f'color: {style.positive}')
        self.neg_button.setStyleSheet(f'color: {style.negative}')
        self._editing = False

    def _map_visibility(self, item):
        if not self._editing:
            self.maps[self.map_list.row(item)].visible = item.checkState() == QtCore.Qt.Checked
            self.refresh()

    def _map_style_changed(self, *args):
        if self._editing:
            return
        index = self.map_list.currentRow()
        if index < 0:
            return
        style = self.maps[index]
        style.alpha = self.map_alpha.value() / 100
        style.palette = self.map_palette.currentText()
        style.low, style.high = self.map_low.value(), self.map_high.value()
        style.threshold = self.map_threshold.value()
        self.map_high.setStyleSheet('' if style.high > style.low else 'background: #8b3434')
        self.refresh()

    def _map_color(self, attr):
        index = self.map_list.currentRow()
        if index < 0:
            return
        style = self.maps[index]
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(getattr(style, attr)), self)
        if color.isValid():
            setattr(style, attr, color.name())
            self._select_map(index)
            self.refresh()

    def _reset_map_range(self):
        index = self.map_list.currentRow()
        if index >= 0:
            self._default_map_range(self.maps[index])
            self._select_map(index)
            self.refresh()

    def _remove_map(self):
        index = self.map_list.currentRow()
        if index >= 0:
            self._editing = True
            self.maps.pop(index)
            self.map_list.takeItem(index)
            self._editing = False
            self._select_map(self.map_list.currentRow())
            self.refresh()

    def _move_map(self, delta):
        index = self.map_list.currentRow()
        target = index + delta
        if 0 <= index < len(self.maps) and 0 <= target < len(self.maps):
            self._editing = True
            self.maps.insert(target, self.maps.pop(index))
            item = self.map_list.takeItem(index)
            self.map_list.insertItem(target, item)
            self._editing = False
            self.map_list.setCurrentRow(target)
            self.refresh()

    def _load_feat(self):
        self.design = self.contrasts = self.residual_layer = None
        self.pe_layers, self.cope_layers = {}, {}
        self.model_warnings = []
        if self.feat_dir:
            for filename, attr in [('design.mat', 'design'), ('design.con', 'contrasts')]:
                path = self.feat_dir / filename
                if path.exists():
                    try:
                        setattr(self, attr, read_fsl_matrix(path))
                    except Exception as exc:
                        self.model_warnings.append(str(exc))
            for prefix, destination in [('pe', self.pe_layers), ('cope', self.cope_layers)]:
                for path in sorted((self.feat_dir / 'stats').glob(prefix + '*.nii*')):
                    match = re.fullmatch(prefix + r'(\d+)\.nii(?:\.gz)?', path.name)
                    if match:
                        try:
                            layer = self._load_nifti_lpi(path, path.name)
                            if not self._same_grid(self.base_layer, layer) or layer.is_4d:
                                raise ValueError(f'{path.name}: incompatible coefficient grid/dimensions')
                            destination[int(match[1])] = layer
                        except Exception as exc:
                            self.model_warnings.append(str(exc))
            path = self._find_image(self.feat_dir / 'stats', 'res4d')
            if path and self.base_layer.path.name in ('filtered_func_data.nii', 'filtered_func_data.nii.gz'):
                try:
                    layer = self._load_nifti_lpi(path, 'residuals')
                    if not self._same_grid(self.base_layer, layer) or layer.shape != self.base_layer.shape:
                        raise ValueError('res4d: incompatible grid or volume count')
                    self.residual_layer = layer
                except Exception as exc:
                    self.model_warnings.append(str(exc))
            if self.design is not None and self.design.shape[0] != self.base_layer.nvolumes:
                self.model_warnings.append('Design rows differ from base volumes; model curves disabled.')
            # FEAT voxelwise EV columns require their per-voxel images. Do not silently use placeholders.
            fsf = self.feat_dir / 'design.fsf'
            self.voxelwise_evs = False
            if fsf.exists():
                text = fsf.read_text(errors='replace')
                self.voxelwise_evs = bool(re.search(r'set\s+fmri\(evs_vox\)\s+"?[1-9]', text))
            if self.voxelwise_evs:
                self.model_warnings.append('Voxelwise EV design detected; design-based curves disabled.')
        else:
            self.voxelwise_evs = False
        count = self.design.shape[1] if self.design is not None else max(self.pe_layers, default=0)
        if self.feat_dir:
            self.ev_names, self.contrast_names, settings, warnings = read_feat_names(self.feat_dir, count)
            self.model_warnings.extend(warnings)
            self.voxelwise_evs = self.voxelwise_evs or any(
                key.startswith('shape') and value == '9' for key, value in settings.items()) or (self.feat_dir / 'ven.dat').exists()
        else:
            self.ev_names, self.contrast_names = {}, {}
        self._populate_design()
        self._populate_traces()
        message = 'No FEAT model loaded.' if not self.feat_dir else (
            f'{len(self.pe_layers)} PE maps; {len(self.cope_layers)} COPE maps.\n'
            'Predictions are in-sample Xβ. COPE partial fits use normalized absolute contrast weights (FSLeyes convention).\n'
            'Raw EVs retain design units. Double-click a trace to choose its color.')
        self.model_info.setText(message + ('\n' + '\n'.join(self.model_warnings) if self.model_warnings else ''))
        self.model_info.setToolTip(str(self.feat_dir or ''))

    def _ev_name(self, number):
        return self.ev_names.get(number, 'Unnamed design column')

    def _contrast_name(self, number):
        return self.contrast_names.get(number, 'Unnamed contrast')

    def _contrast_definition(self, number):
        if self.contrasts is None or not 1 <= number <= len(self.contrasts):
            return 'Contrast weights unavailable'
        weights = self.contrasts[number-1]
        terms = [f'{weight:+g} × PE {i+1} ({self._ev_name(i+1)})'
                 for i, weight in enumerate(weights) if weight != 0]
        return ' '.join(terms).lstrip('+') if terms else '0 (all weights zero)'

    def _map_title(self, path):
        # Only apply this analysis's names to its own stats files.
        if not self.feat_dir or path.parent.resolve() != (self.feat_dir / 'stats').resolve():
            return path.name
        match = re.fullmatch(r'(pe|cope|varcope|tstat|zstat)(\d+)\.nii(?:\.gz)?', path.name)
        if not match:
            return path.name
        kind, number = match[1], int(match[2])
        name = self._ev_name(number) if kind == 'pe' else self._contrast_name(number)
        return f'{path.name} — {name}'

    def _populate_design(self):
        self.ev_table.clear()
        self.contrast_table.clear()
        for number, name in sorted(self.ev_names.items()):
            item = QtWidgets.QTreeWidgetItem([f'PE {number}', name])
            item.setToolTip(0, f'Design column {number}; stats/pe{number}.nii[.gz]')
            item.setToolTip(1, name)
            self.ev_table.addTopLevelItem(item)
        numbers = set(self.contrast_names) | set(self.cope_layers)
        if self.contrasts is not None:
            numbers.update(range(1, len(self.contrasts)+1))
        for number in sorted(numbers):
            definition = self._contrast_definition(number)
            item = QtWidgets.QTreeWidgetItem([str(number), self._contrast_name(number), definition])
            item.setData(0, QtCore.Qt.UserRole, number)
            tooltip = f'Contrast {number}: {self._contrast_name(number)}\n{definition}\n'
            tooltip += f'stats/cope{number}, tstat{number}, zstat{number}, varcope{number}'
            for column in range(3):
                item.setToolTip(column, tooltip)
            self.contrast_table.addTopLevelItem(item)
        for table in (self.ev_table, self.contrast_table):
            table.setColumnWidth(0, 75)
            table.setColumnWidth(1, 190)
        if self.contrast_table.topLevelItemCount():
            self.contrast_table.setCurrentItem(self.contrast_table.topLevelItem(0))
        self._filter_design(self.design_filter.text())

    def _show_contrast_detail(self, item, previous=None):
        if item is None:
            self.contrast_detail.setText('Select a contrast to inspect its definition.')
            return
        number = item.data(0, QtCore.Qt.UserRole)
        self.contrast_detail.setText(f'Contrast {number} — {self._contrast_name(number)}\n'
                                     + self._contrast_definition(number))

    def _filter_design(self, text):
        query = text.casefold().strip()
        for table in (self.ev_table, self.contrast_table):
            for row in range(table.topLevelItemCount()):
                item = table.topLevelItem(row)
                item.setHidden(query not in ' '.join(item.text(c) for c in range(table.columnCount())).casefold())

    def _load_contrast_map(self):
        item = self.contrast_table.currentItem()
        if item is None or not self.feat_dir:
            self.status.setText('Select a contrast in the Design tab first.')
            return
        number = item.data(0, QtCore.Qt.UserRole)
        stem = f'{self.contrast_map_kind.currentText()}{number}'
        path = self._find_image(self.feat_dir / 'stats', stem)
        if path is None:
            self.status.setText(f'{stem}.nii[.gz] is not available for {self._contrast_name(number)}.')
            return
        self.safe_load(self.load_overlay, path)

    def _design_ready(self):
        return (self.base_layer is not None and self.base_layer.is_4d and self.design is not None
                and self.design.shape[0] == self.base_layer.nvolumes and not self.voxelwise_evs
                and self.base_layer.path.name in ('filtered_func_data.nii', 'filtered_func_data.nii.gz'))

    def _populate_traces(self):
        self._editing = True
        self.trace_table.clear()
        def add(key, label, checked=False, available=True):
            item = QtWidgets.QTreeWidgetItem([label, '●'])
            item.setData(0, QtCore.Qt.UserRole, key)
            item.setData(1, QtCore.Qt.UserRole, self.COLORS[self.trace_table.topLevelItemCount() % len(self.COLORS)])
            item.setForeground(1, QtGui.QColor(item.data(1, QtCore.Qt.UserRole)))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.Checked if checked and available else QtCore.Qt.Unchecked)
            item.setToolTip(0, label + ('\n' + self._contrast_definition(int(key.split(':')[1])) if key.startswith(('cope:', 'copefit:')) else ''))
            item.setDisabled(not available)
            if not available:
                item.setToolTip(0, 'Required FEAT model files, matching grid or time points are unavailable.')
            self.trace_table.addTopLevelItem(item)
        ready = self._design_ready()
        full = ready and all(i + 1 in self.pe_layers for i in range(self.design.shape[1]))
        add('bold', 'BOLD signal', True)
        add('fit', 'Fit: data − res4d', True, self.residual_layer is not None)
        add('prediction', 'Prediction: Xβ', False, full)
        add('residual', 'Residuals: res4d', False, self.residual_layer is not None)
        add('selected', 'Prediction: selected PEs', False, ready)
        if self.design is not None:
            for i in range(self.design.shape[1]):
                add(f'ev:{i+1}', f'EV {i+1} — {self._ev_name(i+1)}: design column', available=ready)
                add(f'pefit:{i+1}', f'PE {i+1} — {self._ev_name(i+1)}: fitted contribution', available=ready and i+1 in self.pe_layers)
        for i in sorted(self.pe_layers):
            add(f'pe:{i}', f'PE {i} — {self._ev_name(i)}: coefficient value')
        for i in sorted(self.cope_layers):
            add(f'cope:{i}', f'COPE {i} — {self._contrast_name(i)}: coefficient value')
        if self.contrasts is not None:
            for i, contrast in enumerate(self.contrasts):
                available = (ready and contrast.size == self.design.shape[1]
                             and any(contrast) and all(j+1 in self.pe_layers for j in np.flatnonzero(contrast)))
                add(f'copefit:{i+1}', f'COPE {i+1} — {self._contrast_name(i+1)}: partial fit', available=available)
        self._editing = False

    def _trace_changed(self, *args):
        if not self._editing:
            self._update_time_series()

    def _trace_color(self, item, column):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(item.data(1, QtCore.Qt.UserRole)), self)
        if color.isValid():
            item.setData(1, QtCore.Qt.UserRole, color.name())
            item.setForeground(1, color)
            self._update_time_series()

    def _selected_traces(self):
        return [self.trace_table.topLevelItem(i) for i in range(self.trace_table.topLevelItemCount())
                if self.trace_table.topLevelItem(i).checkState(0) == QtCore.Qt.Checked
                and not self.trace_table.topLevelItem(i).isDisabled()]

    def _collect_series(self):
        if self.base_layer is None:
            return {}, []
        voxel = self.state.voxel
        bold = np.atleast_1d(np.asarray(self.base_layer.data[voxel], dtype=float))
        center = bool(self.demean.isChecked())
        mean = float(np.nanmean(bold)) if np.any(np.isfinite(bold)) else 0.0
        selected = self._selected_traces()
        pe_indices = [int(item.data(0, QtCore.Qt.UserRole).split(':')[1]) for item in selected
                      if item.data(0, QtCore.Qt.UserRole).startswith('pefit:')]
        # Mean-restored design predictions only make sense for first-level time series.
        offset = 0.0 if center else mean
        series, coefficients = {}, []
        for item in selected:
            key = item.data(0, QtCore.Qt.UserRole)
            color = item.data(1, QtCore.Qt.UserRole)
            label = item.text(0)
            values = None
            if key == 'bold':
                values = bold - mean if center else bold
            elif key == 'fit':
                values = bold - np.asarray(self.residual_layer.data[voxel], dtype=float)
                if center:
                    values = values - mean
            elif key == 'residual':
                values = np.asarray(self.residual_layer.data[voxel], dtype=float)
            elif key in ('prediction', 'selected'):
                indices = list(range(1, self.design.shape[1]+1)) if key == 'prediction' else pe_indices
                if indices:
                    values = sum(self.design[:, i-1] * float(self.pe_layers[i].data[voxel]) for i in indices) + offset
            else:
                kind, number = key.split(':')
                number = int(number)
                if kind in ('pe', 'cope'):
                    layers = self.pe_layers if kind == 'pe' else self.cope_layers
                    coefficients.append((label, float(layers[number].data[voxel]), color))
                elif kind == 'ev':
                    values = self.design[:, number-1]
                elif kind == 'pefit':
                    values = self.design[:, number-1] * float(self.pe_layers[number].data[voxel]) + offset
                elif kind == 'copefit':
                    c = self.contrasts[number-1]
                    weights = np.abs(c) * np.sqrt(np.count_nonzero(c)) / np.linalg.norm(c)
                    values = sum(self.design[:, i] * float(self.pe_layers[i+1].data[voxel]) * weights[i]
                                 for i in np.flatnonzero(weights)) + offset
            if values is not None:
                series[key] = (label, np.asarray(values, dtype=float), color)
        return series, coefficients

    def _update_time_series(self, force_autorange=False):
        if self._editing or self.base_layer is None:
            return
        series, coefficients = self._collect_series()
        self._series = series
        for key in list(self._curves):
            if key not in series:
                self.ts_plot.removeItem(self._curves.pop(key))
        for key, (label, values, color) in series.items():
            if key not in self._curves:
                self._curves[key] = self.ts_plot.plot(name=label)
            self._curves[key].setPen(pg.mkPen(color, width=1.8))
            self._curves[key].setData(np.arange(values.size), values, connect='finite')
        self.coef_plot.clear()
        ticks = []
        for i, (label, value, color) in enumerate(coefficients):
            if np.isfinite(value):
                self.coef_plot.addItem(pg.BarGraphItem(x=[i], height=[value], width=0.65, brush=color))
            ticks.append((i, label.rsplit(':', 1)[0]))
        self.coef_plot.getAxis('bottom').setTicks([ticks])
        self.t_marker.setPos(self.state.volume)
        if self.auto_ts_scale.isChecked() or force_autorange:
            self.ts_plot.enableAutoRange()
            self.coef_plot.enableAutoRange()
        else:
            self.ts_plot.disableAutoRange()
            self.coef_plot.disableAutoRange()

    @staticmethod
    def _map_rgba(data, style):
        rgba = np.zeros(data.shape + (4,), dtype=np.uint8)
        if style.high <= style.low:
            return rgba
        valid = np.isfinite(data) & (data != 0) & (np.abs(data) >= style.threshold)
        values = np.where(np.isfinite(data), data, 0)
        if style.palette == 'Signed':
            for mask, color, extent in [(valid & (values > 0), style.positive, max(style.high, 1e-12)),
                                        (valid & (values < 0), style.negative, max(-style.low, 1e-12))]:
                rgb = np.array(QtGui.QColor(color).getRgb()[:3], dtype=float)
                brightness = 0.35 + 0.65 * np.clip(np.abs(values[mask]) / extent, 0, 1)
                rgba[mask, :3] = (brightness[:, None] * rgb).astype(np.uint8)
        else:
            scaled = np.clip((values - style.low) / (style.high - style.low), 0, 1)
            rgba[..., :3] = pg.colormap.get(style.palette).map(scaled, mode='byte')[..., :3]
        rgba[..., 3][valid] = round(255 * style.alpha)
        return rgba

    def refresh(self):
        if self.base_layer is None or self._refreshing:
            return
        self._refreshing = True
        try:
            x, y, z = self.state.voxel
            volume = self.base_layer.volume(self.state.volume)
            slices = self._make_lpi_slices(volume, x, y, z)
            for view in (self.axial, self.coronal, self.sagittal):
                view.set_images(slices[view.orientation], self.state.base_range or DisplayRange(0, 1),
                                None, DisplayRange(-1, 1), 0, 0, 1, -1, 0)
                shape = slices[view.orientation].shape
                rgb = np.zeros(shape + (3,), dtype=float)
                alpha = np.zeros(shape, dtype=float)
                for style in self.maps:
                    if not style.visible:
                        continue
                    plane = self._make_lpi_slices(style.layer.volume(self.state.volume), x, y, z)[view.orientation]
                    rgba = self._map_rgba(plane, style).astype(float) / 255
                    a = rgba[..., 3]
                    rgb = rgba[..., :3] * a[..., None] + rgb * (1-a[..., None])
                    alpha = a + alpha * (1-a)
                out = np.zeros(shape + (4,), dtype=np.uint8)
                out[..., :3] = (np.divide(rgb, alpha[..., None], out=np.zeros_like(rgb), where=alpha[..., None] > 0) * 255).astype(np.uint8)
                out[..., 3] = (alpha * 255).astype(np.uint8)
                view.overlay_item.setImage(out, autoLevels=False)
            sy, sz = self.base_layer.shape[1:3]
            self.axial.set_crosshair((x, sy-1-y))
            self.coronal.set_crosshair((x, sz-1-z))
            self.sagittal.set_crosshair((y, sz-1-z))
            self._update_time_series()
            world = nib.affines.apply_affine(self.base_layer.affine, (x, y, z))
            self.status.setText(f'LPI voxel {x}, {y}, {z}  |  RAS mm {world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f}  |  volume {self.state.volume}  |  value {volume[x,y,z]:.5g}')
        finally:
            self._refreshing = False

    def _save_time_series(self):
        if self.base_layer is None:
            return
        series, coefficients = self._collect_series()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export displayed voxel data',
                    f'voxel_{"_".join(map(str, self.state.voxel))}.csv', 'CSV (*.csv)')
        if not path:
            return
        try:
            n = max([v[1].size for v in series.values()] + [1])
            # Coefficients occupy row zero only; they are scalar estimates, not time courses.
            with open(path, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['volume'] + [v[0] for v in series.values()] + [c[0] + ' (scalar at row 0)' for c in coefficients])
                for t in range(n):
                    writer.writerow([t] + [v[1][t] if t < v[1].size else '' for v in series.values()]
                                    + [c[1] if t == 0 else '' for c in coefficients])
            self.status.setText(f'Saved {path}')
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, 'Export failed', str(exc))
    @staticmethod
    def _load_nifti_lpi(path: Path, name: str) -> NiftiLayer:
        path = Path(path).expanduser().resolve()
        logger.info("Loading %s", path)
        img = nib.load(str(path))
        if len(img.shape) not in (3, 4) or any(n < 1 for n in img.shape):
            raise ValueError(f"Expected nonempty 3D/4D NIfTI; got {img.shape}")
        original_axcodes = nio.aff2axcodes(img.affine)
        img = nib.as_closest_canonical(img, enforce_diag=False)

        # nib.as_closest_canonical returns RAS+. Convert RAS -> LPI by flipping all axes.
        data = np.asanyarray(img.dataobj)
        data = data[::-1, ::-1, ::-1, ...]
        affine = img.affine.copy()
        for axis in range(3):
            affine[:, axis] *= -1
            affine[:, 3] += img.affine[:, axis] * (img.shape[axis] - 1)

        return NiftiLayer(
            path=path,
            img=img,
            data=data,
            affine=affine,
            name=name,
            original_axcodes=tuple(str(a) for a in original_axcodes),
        )


    def _configure_spins(self) -> None:
        if self.base_layer is None:
            return
        sx, sy, sz = self.base_layer.shape[:3]
        self.x_spin.setRange(0, sx - 1)
        self.y_spin.setRange(0, sy - 1)
        self.z_spin.setRange(0, sz - 1)
        self._sync_spins_to_state()


    def center_voxel(self, refresh: bool = True) -> None:
        if self.base_layer is None:
            return
        sx, sy, sz = self.base_layer.shape[:3]
        self.state.voxel = (sx // 2, sy // 2, sz // 2)
        self._sync_spins_to_state()
        if refresh:
            self.refresh()


    def reset_views(self) -> None:
        for view in (self.axial, self.coronal, self.sagittal):
            view.reset_view_range()


    def reset_base_range(self, refresh: bool = True) -> None:
        if self.base_layer is None:
            return
        vol = np.asarray(self.base_layer.volume(self.state.volume), dtype=float)
        finite = vol[np.isfinite(vol)]
        if finite.size == 0:
            rng = DisplayRange(0, 1)
        else:
            rng = DisplayRange(*np.percentile(finite, [1, 99.5]))
        if not rng.valid():
            rng = DisplayRange(rng.vmin, rng.vmin + 1.0)
        self.state.base_range = rng
        self.base_range.set_range(rng.vmin, rng.vmax)
        if refresh:
            self.refresh()


    def _sync_spins_to_state(self) -> None:
        x, y, z = self.state.voxel
        for spin, value in ((self.x_spin, x), (self.y_spin, y), (self.z_spin, z)):
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)


    def _spin_voxel_changed(self) -> None:
        if self._refreshing or self.base_layer is None:
            return
        self.state.voxel = (self.x_spin.value(), self.y_spin.value(), self.z_spin.value())
        self.refresh()


    def _set_volume(self, value: int) -> None:
        if self._refreshing:
            return
        self.state.volume = int(value)
        self.t_slider.blockSignals(True)
        self.t_spin.blockSignals(True)
        self.t_slider.setValue(value)
        self.t_spin.setValue(value)
        self.t_slider.blockSignals(False)
        self.t_spin.blockSignals(False)
        self.refresh()


    def _slice_position_changed(self, payload: tuple) -> None:
        if self.base_layer is None:
            return
        orientation, a, b = payload
        x, y, z = self.state.voxel
        sx, sy, sz = self.base_layer.shape[:3]

        # LPI display mapping:
        # axial image coords:    x=L/R, y=P/A
        # coronal image coords:  x=L/R, y=I/S
        # sagittal image coords: x=P/A, y=I/S
        if orientation == "axial":
            x, y = a, sy - 1 - b
        elif orientation == "coronal":
            x, z = a, sz - 1 - b
        elif orientation == "sagittal":
            y, z = a, sz - 1 - b

        self.state.voxel = (
            int(np.clip(x, 0, sx - 1)),
            int(np.clip(y, 0, sy - 1)),
            int(np.clip(z, 0, sz - 1)),
        )
        self._sync_spins_to_state()
        self.refresh()


    @staticmethod
    def _make_lpi_slices(vol: Optional[Array], x: int, y: int, z: int) -> Dict[str, Array]:
        if vol is None:
            return {}
        # Data axes are canonical L, P, I. pyqtgraph image rows are vertical.
        # Display is flipped around the horizontal/x axis so that the anatomical
        # inferior/posterior direction is shown at the bottom of the pane.
        return {
            "axial": np.flipud(vol[:, :, z].T),       # horizontal L/R, vertical P/A, P at bottom
            "coronal": np.flipud(vol[:, y, :].T),     # horizontal L/R, vertical I/S, I at bottom
            "sagittal": np.flipud(vol[x, :, :].T),    # horizontal P/A, vertical I/S, I at bottom
        }


    def _range_controls_changed(self):
        rng = self.base_range.range()
        if rng.valid():
            self.state.base_range = rng
            self.refresh()
        self.base_range.max_spin.setStyleSheet('' if rng.valid() else 'background: #8b3434')


def main(argv=None):
    import argparse
    argv = sys.argv if argv is None else argv
    parser = argparse.ArgumentParser(description='BOLD / FEAT viewer')
    parser.add_argument('image', nargs='?', type=Path, help='3D/4D NIfTI or FEAT directory')
    parser.add_argument('-o', '--overlay', type=Path, action='append', default=[], help='Overlay map; repeat for multiple maps')
    args = parser.parse_args(argv[1:])
    pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv[:1])
    app.setApplicationName('BOLD Viewer')
    viewer = BoldViewer(args.image)
    for path in args.overlay:
        viewer.safe_load(viewer.load_overlay, path)
    viewer.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
