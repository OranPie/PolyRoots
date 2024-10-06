import wx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from itertools import product
from multiprocessing import Pool, cpu_count
import os
import threading
from matplotlib.colors import LogNorm
import logging

# 配置日志
logging.basicConfig(
    filename='polyroots.log',
    filemode='a',  # 追加模式
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

# 使用WXAgg后端
matplotlib.use('WXAgg')

# 热图范围常量
X_MAX = 1.8
X_MIN = -1.8
Y_MAX = 1.8
Y_MIN = -1.8


def compute_roots(poly_coeffs):
    """计算单个多项式的根。"""
    try:
        roots = np.roots(poly_coeffs)
        logging.debug(f"Computed roots for polynomial coefficients: {poly_coeffs}")
        return roots
    except Exception as e:
        logging.error(f"Error computing roots for coefficients {poly_coeffs}: {e}")
        return np.array([])


def save_roots(degree, datafile, progress_callback=None):
    """计算并保存所有可能系数为-1或1的多项式的根。"""
    try:
        total_polys = 2 ** degree
        logging.info(f"Starting to save roots for degree {degree}, total polynomials: {total_polys}")

        # 使用生成器表达式避免一次性生成所有组合
        coeff_combinations = ((1,) + combo for combo in product([-1, 1], repeat=degree))

        roots = []
        processed = 0

        # 使用多进程加速计算
        with Pool(processes=cpu_count()) as pool:
            for roots_batch in pool.imap(compute_roots, coeff_combinations):
                roots.append(roots_batch)
                processed += 1
                if progress_callback:
                    progress_callback(processed, total_polys)
                if processed % 10000 == 0:
                    logging.info(f"Processed {processed}/{total_polys} polynomials")

        roots = np.concatenate(roots)
        np.save(datafile, roots)
        logging.info(f"Successfully saved roots to {datafile}")
    except Exception as e:
        logging.error(f"Error in save_roots: {e}")


def heat_map(size, datafile, progress_callback=None):
    """生成热图数据。使用numpy.histogram2d提高效率。"""
    try:
        logging.info(f"Starting heatmap generation with size {size} from datafile {datafile}")
        f = np.load(datafile, 'r')
        x = f.real
        y = f.imag

        # 使用histogram2d计算二维直方图
        img, xedges, yedges = np.histogram2d(x, y, bins=size, range=[[X_MIN, X_MAX], [Y_MIN, Y_MAX]])

        if progress_callback:
            # 假设histogram2d是一个快速操作，此处直接设置为完成
            progress_callback(size, size)

        # 对数缩放来增强对比度
        img = np.log(img + 1)  # 避免log(0)

        logging.debug(f"Heatmap generated with shape {img.shape}")
        return img
    except Exception as e:
        logging.error(f"Error in heat_map: {e}")
        return np.array([])


class PolyRootFrame(wx.Frame):
    def __init__(self, parent, title):
        super(PolyRootFrame, self).__init__(parent, title=title, size=(1000, 800))

        self.InitUI()
        self.Centre()

    def InitUI(self):
        panel = wx.Panel(self)

        # 使用BoxSizer布局
        vbox = wx.BoxSizer(wx.VERTICAL)

        # 控件部分
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        lbl_degree = wx.StaticText(panel, label="Polynomial Degree:")
        hbox1.Add(lbl_degree, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        self.txt_degree = wx.TextCtrl(panel)
        self.txt_degree.SetValue("3")
        hbox1.Add(self.txt_degree, proportion=1)

        lbl_size = wx.StaticText(panel, label="Image Size:")
        hbox1.Add(lbl_size, flag=wx.RIGHT | wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=8)
        self.txt_size = wx.TextCtrl(panel)
        self.txt_size.SetValue("800")
        hbox1.Add(self.txt_size, proportion=1)

        vbox.Add(hbox1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        # DPI控制
        hbox_dpi = wx.BoxSizer(wx.HORIZONTAL)
        lbl_dpi = wx.StaticText(panel, label="DPI:")
        hbox_dpi.Add(lbl_dpi, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        self.txt_dpi = wx.TextCtrl(panel)
        self.txt_dpi.SetValue("300")  # 默认DPI值
        hbox_dpi.Add(self.txt_dpi, proportion=1)
        vbox.Add(hbox_dpi, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        lbl_colormap = wx.StaticText(panel, label="Color Map:")
        hbox2.Add(lbl_colormap, flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=8)
        self.cmb_colormap = wx.ComboBox(panel, choices=sorted([
            'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap',
            'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r',
            'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
            'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r',
            'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r',
            'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r',
            'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn',
            'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
            'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r',
            'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r',
            'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray',
            'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
            'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot',
            'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r',
            'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r',
            'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow',
            'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r',
            'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo',
            'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis',
            'viridis_r', 'winter', 'winter_r'
        ]), style=wx.CB_READONLY)
        self.cmb_colormap.SetValue("afmhot")
        hbox2.Add(self.cmb_colormap, proportion=1)

        lbl_output = wx.StaticText(panel, label="Output Filename:")
        hbox2.Add(lbl_output, flag=wx.RIGHT | wx.LEFT | wx.ALIGN_CENTER_VERTICAL, border=8)
        self.txt_output = wx.TextCtrl(panel)
        self.txt_output.SetValue("polyroots.png")
        hbox2.Add(self.txt_output, proportion=1)

        vbox.Add(hbox2, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_save = wx.Button(panel, label="Roots")
        self.btn_save.Bind(wx.EVT_BUTTON, self.OnSaveRoots)
        hbox3.Add(self.btn_save, flag=wx.RIGHT, border=8)

        self.btn_generate = wx.Button(panel, label="Heatmap")
        self.btn_generate.Bind(wx.EVT_BUTTON, self.OnGenerateHeatmap)
        hbox3.Add(self.btn_generate, flag=wx.RIGHT, border=8)

        self.btn_save_and_generate = wx.Button(panel, label="All")
        self.btn_save_and_generate.Bind(wx.EVT_BUTTON, self.OnSaveAndGenerate)
        hbox3.Add(self.btn_save_and_generate, flag=wx.RIGHT, border=8)

        self.btn_display = wx.Button(panel, label="Display")
        self.btn_display.Bind(wx.EVT_BUTTON, self.OnDisplay)
        hbox3.Add(self.btn_display, flag=wx.RIGHT, border=8)

        vbox.Add(hbox3, flag=wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, border=10)

        # 进度条
        self.gauge = wx.Gauge(panel, range=100, size=(250, 25))
        vbox.Add(self.gauge, flag=wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, border=10)

        # Matplotlib Figure
        self.figure = plt.figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvas(panel, -1, self.figure)
        vbox.Add(self.canvas, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)

        panel.SetSizer(vbox)

    def OnSaveRoots(self, event):
        degree = self.GetDegree()
        if degree is None:
            return
        datafile = f"roots_degree_{degree}.npy"
        dlg = wx.MessageDialog(self, f"Roots将保存到 {datafile}。继续吗？", "确认", wx.OK | wx.CANCEL)
        if dlg.ShowModal() == wx.ID_OK:
            dlg.Destroy()
            self.gauge.SetValue(0)
            # 在后台线程执行以保持GUI响应
            threading.Thread(target=self.RunSaveRoots, args=(degree, datafile), daemon=True).start()
        else:
            dlg.Destroy()

    def OnGenerateHeatmap(self, event):
        degree = self.GetDegree()
        if degree is None:
            return
        size = self.GetSizeValue()
        if size is None:
            return
        datafile = f"roots_degree_{degree}.npy"
        if not os.path.exists(datafile):
            dlg = wx.MessageDialog(self, f"数据文件 {datafile} 不存在。请先保存根。", "错误", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return
        color_map = self.cmb_colormap.GetValue()
        output = self.txt_output.GetValue()
        dpi = self.GetDPI()
        if dpi is None:
            return
        self.gauge.SetValue(0)
        # 在后台线程执行以保持GUI响应
        threading.Thread(target=self.RunGenerateHeatmap, args=(size, datafile, color_map, output, dpi),
                         daemon=True).start()

    def OnDisplay(self, event):
        wx.CallAfter(self.Display)

    def Display(self):
        if hasattr(self, 'img') and self.img.size > 0:
            self.DisplayHeatmap(self.img, self.cmb_colormap.GetValue())
        else:
            dlg = wx.MessageDialog(self, "请先生成热图。", "错误", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()

    def OnSaveAndGenerate(self, event):
        degree = self.GetDegree()
        if degree is None:
            return
        size = self.GetSizeValue()
        if size is None:
            return
        datafile = f"roots_degree_{degree}.npy"
        color_map = self.cmb_colormap.GetValue()
        output = self.txt_output.GetValue()
        dpi = self.GetDPI()
        if dpi is None:
            return
        dlg = wx.MessageDialog(self, f"Roots将保存到 {datafile} 并生成热图。继续吗？", "确认", wx.OK | wx.CANCEL)
        if dlg.ShowModal() == wx.ID_OK:
            dlg.Destroy()
            self.gauge.SetValue(0)
            # 在后台线程执行以保持GUI响应
            threading.Thread(target=self.RunSaveAndGenerate, args=(degree, size, datafile, color_map, output, dpi),
                             daemon=True).start()
        else:
            dlg.Destroy()

    def GetDegree(self):
        try:
            degree = int(self.txt_degree.GetValue())
            if degree < 1 or degree > 25:
                # 增加最大次数上限，视系统内存和处理能力而定
                raise ValueError("Degree out of allowed range (1-25).")
            logging.debug(f"User input degree: {degree}")
            return degree
        except ValueError as e:
            logging.error(f"Invalid degree input: {self.txt_degree.GetValue()} - {e}")
            dlg = wx.MessageDialog(self, "请输入一个有效的正整数作为多项式次数（1-25）。", "错误", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return None

    def GetSizeValue(self):
        try:
            size = int(self.txt_size.GetValue())
            if size < 100 or size > 10000:
                raise ValueError("Size out of allowed range (100-10000).")
            logging.debug(f"User input size: {size}")
            return size
        except ValueError as e:
            logging.error(f"Invalid size input: {self.txt_size.GetValue()} - {e}")
            dlg = wx.MessageDialog(self, "请输入一个有效的图像大小（100-10000）。", "错误", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return None

    def GetDPI(self):
        try:
            dpi = int(self.txt_dpi.GetValue())
            if dpi < 50 or dpi > 10000:
                raise ValueError("DPI out of allowed range (50-10000).")
            logging.debug(f"User input DPI: {dpi}")
            return dpi
        except ValueError as e:
            logging.error(f"Invalid DPI input: {self.txt_dpi.GetValue()} - {e}")
            dlg = wx.MessageDialog(self, "请输入一个有效的DPI值（50-10000）。", "错误", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return None

    def RunSaveRoots(self, degree, datafile):
        try:
            def progress_callback(current, total):
                percent = int((current / total) * 100)
                wx.CallAfter(self.gauge.SetValue, percent)
                logging.debug(f"Saving roots progress: {percent}%")

            save_roots(degree, datafile, progress_callback)
            dlg = wx.MessageDialog(self, f"Roots已保存到 {datafile}。", "信息", wx.OK | wx.ICON_INFORMATION)
            wx.CallAfter(dlg.ShowModal)
            wx.CallAfter(dlg.Destroy)
        except Exception as e:
            logging.error(f"Error in RunSaveRoots: {e}")
            dlg = wx.MessageDialog(self, f"保存根时发生错误: {e}", "错误", wx.OK | wx.ICON_ERROR)
            wx.CallAfter(dlg.ShowModal)
            wx.CallAfter(dlg.Destroy)

    def RunGenerateHeatmap(self, size, datafile, color_map, output, dpi):
        try:
            def progress_callback(current, total):
                percent = int((current / total) * 100)
                wx.CallAfter(self.gauge.SetValue, percent)
                logging.debug(f"Generating heatmap progress: {percent}%")

            self.img = heat_map(size, datafile, progress_callback)
            if self.img.size == 0:
                raise ValueError("生成的热图数据为空。")
            wx.CallAfter(self.DisplayHeatmap, self.img, color_map)
            plt.savefig(output, dpi=dpi)
            logging.info(f"Heatmap saved to {output} with DPI={dpi}")
            dlg = wx.MessageDialog(self, f"热图已保存到 {output}。", "信息", wx.OK | wx.ICON_INFORMATION)
            wx.CallAfter(dlg.ShowModal)
            wx.CallAfter(dlg.Destroy)
        except Exception as e:
            logging.error(f"Error in RunGenerateHeatmap: {e}")
            dlg = wx.MessageDialog(self, f"生成热图时发生错误: {e}", "错误", wx.OK | wx.ICON_ERROR)
            wx.CallAfter(dlg.ShowModal)
            dlg.Destroy()

    def RunSaveAndGenerate(self, degree, size, datafile, color_map, output, dpi):
        try:
            def progress_callback_save(current, total):
                percent = int((current / total) * 50)
                wx.CallAfter(self.gauge.SetValue, percent)
                logging.debug(f"Saving roots progress: {percent}%")

            save_roots(degree, datafile, progress_callback_save)

            def progress_callback_heat(current, total):
                percent = 50 + int((current / total) * 50)
                wx.CallAfter(self.gauge.SetValue, percent)
                logging.debug(f"Generating heatmap progress: {percent}%")

            img = heat_map(size, datafile, progress_callback_heat)
            if img.size == 0:
                raise ValueError("生成的热图数据为空。")
            self.DisplayHeatmap(img, color_map)
            plt.savefig(output, dpi=dpi)

            logging.info(f"Heatmap saved to {output} with DPI={dpi}")
            dlg = wx.MessageDialog(self, f"Roots已保存到 {datafile} 并且热图已保存到 {output}。", "信息",
                                   wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
        except Exception as e:
            logging.error(f"Error in RunSaveAndGenerate: {e}")
            dlg = wx.MessageDialog(self, f"保存根或生成热图时发生错误: {e}", "错误", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()

    def DisplayHeatmap(self, img, color_map):
        try:
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            ax.axis('off')
            if np.any(img > 0):
                # 使用LogNorm进行颜色归一化，避免因大数据量导致的全黑
                norm = LogNorm(vmin=np.percentile(img[img > 0], 1), vmax=img.max())
                im = ax.imshow(img, cmap=color_map, extent=(X_MIN, X_MAX, Y_MIN, Y_MAX), norm=norm,
                               interpolation='nearest')
                logging.debug("Heatmap displayed successfully.")
            else:
                logging.warning("Heatmap data contains no positive values.")
            self.canvas.draw()
        except Exception as e:
            logging.error(f"Error in DisplayHeatmap: {e}")
            dlg = wx.MessageDialog(self, f"显示热图时发生错误: {e}", "错误", wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()


class PolyRootApp(wx.App):
    def OnInit(self):
        frame = PolyRootFrame(None, title="Polynomial Roots Heatmap")
        frame.Show()
        return True


if __name__ == '__main__':
    app = PolyRootApp()
    app.MainLoop()
