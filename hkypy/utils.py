# Author: 赩林, xilin0x7f@163.com
import numpy as np
import pandas as pd

def get_motion(output_path, files_path):
    motion_all = []
    for file_idx, file_path in enumerate(files_path):
        motions = pd.read_csv(file_path, delimiter='\t')
        mean_fd = motions['framewise_displacement'].mean()
        max_rot = max([motions[f'rot_{direction}'].abs().max() for direction in ('x', 'y', 'z')]) * 180.0 / np.pi
        max_trans = max([motions[f'trans_{direction}'].abs().max() for direction in ('x', 'y', 'z')])
        motion_all.append([file_path, mean_fd, max_rot, max_trans])

    motion_all = pd.DataFrame(motion_all,  columns=['file_path', 'mean_fd', 'max_rot', 'max_trans'])

    motion_all.to_csv(output_path, index=False)

def get_matrix_tril(out_path, diag=False, matrix_paths=None):
    res = []
    for matrix_path in matrix_paths:
        matrix = np.loadtxt(matrix_path)
        res.append(matrix[np.triu_indices_from(matrix, k=0 if diag else 1)])

    np.savetxt(out_path, np.array(res))

def matrix_tril_to_matrix(tril_path, diag=False, out_prefix='matrix'):
    tril = np.loadtxt(tril_path)
    if tril.ndim == 1:
        tril = tril.reshape(1, -1)
    width = np.ceil(np.log10(tril.shape[0])).astype(int) + 1
    shape = np.round(0.5 + np.sqrt(1 + 8 * tril.shape[1])/2).astype(int)
    shape = shape - 1 if diag else shape
    for i in range(tril.shape[0]):
        matrix = np.zeros((shape, shape))
        matrix[np.triu_indices_from(matrix, k=0 if diag else 1)] = tril[i, :]
        matrix = matrix + matrix.T
        matrix = matrix - np.diag(np.diag(matrix))/2 if diag else matrix
        np.savetxt(f'{out_prefix}{i+1:0{width}d}.txt', matrix)

def kde_estimate_mode(data, bw='normal_reference', bins=50, out_prefix='kde', ignore=None):
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    from scipy.signal import argrelextrema

    # for local minima
    # argrelextrema(x, np.less)
    kde = sm.nonparametric.KDEUnivariate(data)
    try:
        bw = float(bw)
    except ValueError:
        pass
    kde.fit(kernel='gau', bw=bw)
    density = kde.density
    x_kde = kde.support
    local_maxima = argrelextrema(density, np.greater)[0]
    # print(local_maxima)
    # print(x_kde[local_maxima])
    # 如果有ignore参数并且可以则用略过前n个的极大值点后的极值点占的最大值，如果不能ignore则用众数
    if ignore is not None and len(local_maxima) > ignore:
        mode = x_kde[local_maxima[ignore:][np.argmax(density[local_maxima[ignore:]])]]
    else:
        mode = x_kde[np.argmax(density)]

    np.savetxt(f'{out_prefix}mode.txt', mode[np.newaxis])

    plt.plot(x_kde, density, label='KDE')
    plt.axvline(mode, color='red', linestyle='--', label=f'Mode: {mode:.2f}')
    plt.hist(data, bins=bins, density=True, alpha=0.5, color='grey', label='Data Histogram')
    plt.legend()
    plt.savefig(f'{out_prefix}kde.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    return mode

def make_wordcloud(path, out_path, name='name', freq='freq', rev=False, top=3, ndigits=3, fontsize=12):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    weight_df = pd.read_csv(path)

    word_weights = dict(zip(weight_df[name], 1 / weight_df[freq] if rev else weight_df[freq]))

    wordcloud = WordCloud(
        width=8000, height=6000, background_color='white', random_state=0
    ).generate_from_frequencies(word_weights)

    top_words = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)[:top]
    top_words_text = ", ".join([f"{word}: {round(1 / freq if rev else freq, ndigits)}" for word, freq in top_words])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    axins = inset_axes(ax, width="100%", height="15%", loc='lower center', borderpad=-5)

    axins.axis('off')
    axins.set_facecolor((0, 0, 0, 0))

    axins.annotate(
        top_words_text, xy=(0.5, 0.5), xycoords='axes fraction', fontsize=fontsize, ha='center', va='center', color='black'
    )

    rect = plt.Rectangle((0, 0.25), 1, 0.5, transform=axins.transAxes,
                         fill=False, edgecolor='black', linestyle='dashed')

    axins.add_patch(rect)

    fig.savefig(out_path, format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
