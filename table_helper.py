import latextable
from texttable import Texttable
def build_latex_table(rows, caption, label):

    table_1 = Texttable()
    col_aligns = ['c' for _ in range(len(rows[0])-1)]
    table_1.set_cols_align(["l"]+col_aligns)
    table_1.set_deco(Texttable.HEADER )
    table_1.add_rows(rows)

    print(table_1.draw())
    print('\nLatextable Output:')
    print(latextable.draw_latex(
        table_1, caption=caption, label=label))

