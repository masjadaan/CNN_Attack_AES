# coding: utf-8

# In[3]:


# !/usr/bin/python3
import sqlite3 as sql
import sys
import numpy as np
from matplotlib import pyplot as plt
import h5py


###############################################################################
# library to access and modify the traces database
#
# lib relies on 'samples' being the last column of the trace table,
# the order of all other columns should not matter

# There is conversion from int to BLOB (Binary Large Object)
# 2^8 = 256 --> 256 values each has a corresponding value in BLOB data type
###############################################################################

def to_hex(b):
    return ''.join(format(x, '02x') for x in b)


###############################################################################
# global variables
###############################################################################

table_columns = []


# In[4]:


###############################################################################
# database connection
###############################################################################

def open_db(filename):
    """
    opens connection to the database.
    :param str filename: filename of the database
    :rtype sqlite3.Connection
    """
    con = sql.connect(filename)
    cur = con.cursor()
    cur.execute('SELECT * FROM traces LIMIT 1')
    global table_columns
    table_columns = list(map(lambda x: x[0], cur.description))
    return con


def close_db(con):
    """
    closes connection to the database
    :param sqlite3.Connection con: database connection
    """
    con.close()


# In[5]:


def get_nr_of_entries(db, tablename):
    """
    Returns the number of entries in a table (= number of rows)
    :param sqlite3.Connection db: open database connection
    :param string tablename: Name of the table that should be used
    :rtype The number of entries as integer
    """
    assert isinstance(tablename, str)
    cur = db.cursor()
    cur.execute('SELECT Count(*) FROM ' + str(tablename))
    return cur.fetchone()[0]


def get_nr_of_entries_by_tile(db, tile_x, tile_y):
    """
    Returns the number of entries in a table (= number of rows)
    :param sqlite3.Connection db: open database connection
    :param string tablename: Name of the table that should be used
    :rtype The number of entries as integer
    """
    cur = db.cursor()
    cur.execute('SELECT COUNT(*) FROM traces ' +
                'WHERE tile_x=? AND tile_y=?'
                , (tile_x, tile_y))
    return cur.fetchone()[0]


def blob_to_np_array(blob):
    """
    convert SQLite blob data type to numpy array
    :param blob: blob format
    :rtype numpy.array
    """
    if blob is None:
        return None
    else:
        return np.fromstring(blob, dtype=np.uint8)


def get_trace_iterator(cur):
    """
    Generator function to fetch traces from cursor. A select call on the
    cursor must preceed this operation. Returns iterator over the traces yielding
    numpy arrays.
    :param sqlite3.Cursor cur: cursor with available results
    :rtype iterator over numpy.array
    """
    row = cur.fetchone()
    while row is not None:
        yield blob_to_np_array(row[0])
        # yield np.fromstring(row[0], dtype=np.uint8)
        row = cur.fetchone()


def get_trace(cur):
    """
    Fetches one trace from cursor. A select call on the cursor must preceed this
    operation. Returns numpy array or None if no result is available.
    :param sqlite3.Cursor cur: cursor with available results
    :rtype numpy.array
    """
    row = cur.fetchone()
    if row == None:
        return None
    # return np.fromstring(row[0], dtype=np.uint8)
    return blob_to_np_array(row[0])


def select_traces(db):
    """
    Makes a query to the database and selects all traces.
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces')
    return cur


def select_traces_by_tile(db, tile_x, tile_y):
    """
    Makes a query to the database and selects all traces at one tile
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                'WHERE tile_x=? and tile_y=?', (tile_x, tile_y))
    return cur


def select_all_by_tile(db, tile_x, tile_y):
    """
    Makes a query to the database and selects all traces at one tile
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                'WHERE tile_x=? and tile_y=?', (tile_x, tile_y))
    return cur


def select_all_by_tile_limitrange(db, tile_x, tile_y, num_traces):
    """
    Makes a query to the database and selects all traces at one tile
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                'WHERE tile_x=? and tile_y=?' +
                '   AND trace_id IN (' +
                '       SELECT trace_id' +
                '       FROM traces' +
                '       WHERE tile_x = ? and tile_y = ?' +
                '       LIMIT ?)', (tile_x, tile_y, tile_x, tile_y, num_traces))
    return cur


def select_all_by_tile_ptxt(db, tile_x, tile_y, sbox, ptxt):
    """
    Makes a query to the database and selects traces and headers at one tile
    with a specific plaintext byte value at a given position
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int ptxt: Plaintext input value of the s-box
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' WHERE pconv.int_val = ?' +
                '   AND tile_x = ? AND tile_y = ?'
                , (sbox + 1, ptxt, tile_x, tile_y))
    return cur


def select_traces_by_tile_ptxt(db, tile_x, tile_y, sbox, ptxt):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific plaintext byte value at a given position
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int ptxt: Plaintext input value of the s-box
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' WHERE pconv.int_val = ?' +
                '   AND tile_x = ? AND tile_y = ?'
                , (sbox + 1, ptxt, tile_x, tile_y))
    return cur


def select_traces_by_tile_ptxt_limit(db, tile_x, tile_y, sbox, ptxt, limit):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific plaintext byte value at a given position
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int ptxt: Plaintext input value of the s-box
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' WHERE pconv.int_val = ?' +
                '   AND tile_x = ? AND tile_y = ?' +
                ' LIMIT ?'
                , (sbox + 1, ptxt, tile_x, tile_y, limit))
    return cur


def select_traces_by_tile_not_ptxt(db, tile_x, tile_y, sbox, ptxt):
    """
    Makes a query to the database and selects all traces at one tile with all but one
    specific plaintext byte value at a given position
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int ptxt: Plaintext input value of the s-box
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' WHERE pconv.int_val != ?' +
                '   AND tile_x = ? AND tile_y = ?'
                , (sbox + 1, ptxt, tile_x, tile_y))
    return cur


def select_traces_by_tile_not_ptxt_limit(db, tile_x, tile_y, sbox, ptxt, limit):
    """
    Makes a query to the database and selects all traces at one tile with all but one
    specific plaintext byte value at a given position
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int ptxt: Plaintext input value of the s-box
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' WHERE pconv.int_val != ?' +
                '   AND tile_x = ? AND tile_y = ?' +
                ' LIMIT ?'
                , (sbox + 1, ptxt, tile_x, tile_y, limit))
    return cur


def select_traces_by_tile_ctxt(db, tile_x, tile_y, sbox, ctxt):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific plaintext byte value at a given position
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int ctxt: ciphertext input value of the s-box
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ctxt,?,1) = pconv.blob_val' +
                ' WHERE pconv.int_val = ?' +
                '   AND tile_x = ? AND tile_y = ?'
                , (sbox + 1, ctxt, tile_x, tile_y))
    return cur


def select_traces_by_tile_ctxt_limit(db, tile_x, tile_y, sbox, ctxt, limit):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific plaintext byte value at a given position
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int ctxt: ciphertext input value of the s-box
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ctxt,?,1) = pconv.blob_val' +
                ' WHERE pconv.int_val = ?' +
                '   AND tile_x = ? AND tile_y = ?' +
                ' LIMIT ?'
                , (sbox + 1, ctxt, tile_x, tile_y, limit))
    return cur


def select_traces_by_tile_not_ctxt(db, tile_x, tile_y, sbox, ctxt):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific plaintext byte value at a given position
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int ctxt: ciphertext input value of the s-box
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ctxt,?,1) = pconv.blob_val' +
                ' WHERE pconv.int_val != ?' +
                '   AND tile_x = ? AND tile_y = ?'
                , (sbox + 1, ctxt, tile_x, tile_y))
    return cur


def select_traces_by_tile_not_ctxt_limit(db, tile_x, tile_y, sbox, ctxt, limit):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific plaintext byte value at a given position
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int ctxt: ciphertext input value of the s-box
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ctxt,?,1) = pconv.blob_val' +
                ' WHERE pconv.int_val != ?' +
                '   AND tile_x = ? AND tile_y = ?' +
                ' LIMIT ?'
                , (sbox + 1, ctxt, tile_x, tile_y, limit))
    return cur


def select_all_by_tile_sboxin(db, tile_x, tile_y, sbox, sboxin):
    """
    Makes a query to the database and selects traces and headers at one tile
    with a specific input value at a given s-box
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(k,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' wHERE (~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) = ?' +
                '   AND tile_x = ? AND tile_y = ?'
                , (sbox + 1, sbox + 1, sboxin, tile_x, tile_y))
    return cur


def select_traces_by_tile_two_sboxin(db, tile_x, tile_y, sbox, sboxin, sbox2, sboxin2):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific input value of 2 given s-boxes
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :param int sbox2: Number of the 2nd s-box (equals byte number)
    :param int sboxin2: Input value of the 2nd s-box (keybyte xor plaintext byte)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(k,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' JOIN byte_conv kconv2' +
                '   ON substr(k,?,1) = kconv2.blob_val' +
                ' JOIN byte_conv pconv2' +
                '   ON substr(ptxt,?,1) = pconv2.blob_val' +
                ' WHERE (((~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) = ?) AND ((~(kconv2.int_val&pconv2.int_val))&(kconv2.int_val|pconv2.int_val) = ?))' +  # XOR between (k and ptxt) AND (k2 and ptxt2)
                '   AND tile_x = ? AND tile_y = ?'
                , (sbox + 1, sbox + 1, sbox2 + 1, sbox2 + 1, sboxin, sboxin2, tile_x, tile_y))
    return cur


def select_traces_by_tile_sboxin(db, tile_x, tile_y, sbox, sboxin):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific input value at a given s-box
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(k,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' wHERE (~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) = ?' +  # XOR between k and ptxt
                '   AND tile_x = ? AND tile_y = ?'
                , (sbox + 1, sbox + 1, sboxin, tile_x, tile_y))
    return cur


roundmapping = {16: 10, 24: 12, 32: 14}
sr_index_translation = [1, 6, 11, 16, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12]
sbox_lut = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]


def select_all_by_tile_last_sboxin(db, tile_x, tile_y, sbox, sboxin, keylength):
    """
    Makes a query to the database and selects traces and headers at one tile
    with a specific output value after the last AES round at a given s-box
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """

    try:
        rounds = roundmapping[keylength]
    except KeyError:
        print("Key length " + str(keylength) + " is not supported. Valid lengths are 16, 24, 32")
        sys.exit(1)

    exp_key_len = (rounds + 1) * 16
    idx_after_shiftrows = sr_index_translation[sbox]
    key_idx = exp_key_len - (16 - idx_after_shiftrows)
    # print('sbox nr: '+ str(sbox))
    # print('ctxt-idx' + str(idx_after_shiftrows))
    # print('key-idx' + str(key_idx))

    sboxout = sbox_lut[sboxin]
    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(adata,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ctxt,?,1) = pconv.blob_val' +
                ' WHERE (~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) = ?' +
                '   AND tile_x = ? AND tile_y = ?'
                , (key_idx, idx_after_shiftrows, sboxout, tile_x, tile_y))
    return cur


def select_traces_by_tile_last_sboxin(db, tile_x, tile_y, sbox, sboxin, keylength):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific output value after the last AES round at a given s-box
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    try:
        rounds = roundmapping[keylength]
    except KeyError:
        print("Key length " + str(keylength) + " is not supported. Valid lengths are 16, 24, 32")
        sys.exit(1)

    sboxout = sbox_lut[sboxin]
    exp_key_len = (rounds + 1) * 16
    idx_after_shiftrows = sr_index_translation[sbox]
    key_idx = exp_key_len - (16 - idx_after_shiftrows)
    # print('sbox nr: '+ str(sbox))
    # print('ctxt-idx' + str(idx_after_shiftrows))
    # print('key-idx' + str(key_idx))

    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(adata,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ctxt,?,1) = pconv.blob_val' +
                ' WHERE (~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) = ?' +
                '   AND tile_x = ? AND tile_y = ?'
                , (key_idx, idx_after_shiftrows, sboxout, tile_x, tile_y))
    return cur


def select_all_by_tile_sboxin_limitrange(db, tile_x, tile_y, sbox, sboxin, num_traces):
    """
    Makes a query to the database and selects traces and headers at one tile
    with a specific input value at a given S-box.
    Only searches the first num_traces entries at that tile
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :param int num_entries: Number of entries which are searched (use to artificially limit the number of traces)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(k,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' wHERE (~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) = ?' +
                '   AND tile_x = ? AND tile_y = ?' +
                '   AND trace_id IN (' +
                '       SELECT trace_id' +
                '       FROM traces' +
                '       WHERE tile_x = ? and tile_y = ?' +
                '       LIMIT ?)'
                , (sbox + 1, sbox + 1, sboxin, tile_x, tile_y, tile_x, tile_y, num_traces))
    return cur


def select_traces_by_tile_sboxin_limitrange(db, tile_x, tile_y, sbox, sboxin, num_traces):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific input value at a given s-box
    Only searches the first num_traces entries at that tile
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :param int num_entries: Number of entries which are searched (use to artificially limit the number of traces)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(k,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' wHERE (~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) = ?' +  # XOR between k and ptxt
                '   AND tile_x = ? AND tile_y = ?' +
                '   AND trace_id IN (' +
                '       SELECT trace_id' +
                '       FROM traces' +
                '       WHERE tile_x = ? and tile_y = ?' +
                '       LIMIT ?)'
                , (sbox + 1, sbox + 1, sboxin, tile_x, tile_y, tile_x, tile_y, num_traces))
    return cur


def select_traces_by_tile_sboxin_second_order_masking_limitrange(db, tile_x, tile_y, sbox, sboxin, mask1_name,
                                                                 mask2_name, mask1_pos, mask2_pos, num_traces):
    # VERY SLOW!!! NOT (funtionally) TESTED!!!!
    # Better use code example in template creation for multiprobe to select the traces (attacktool/main/attacks/ta.py; function aes_multiprobe_create_trace_selection_func)
    # Code example is ugly, but it works
    """
    Makes a query to the database and selects all traces at one tile with a
    specific input value at a given s-box
    Only searches the first num_traces entries at that tile
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :param int num_entries: Number of entries which are searched (use to artificially limit the number of traces)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    raise NotImplementedError
    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(k,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +

                ' JOIN byte_conv m1conv' +
                '   ON substr(' + mask1_name + ',?,1) = kconv.blob_val' +

                ' JOIN byte_conv m2conv' +
                '   ON substr(' + mask2_name + ',?,1) = pconv.blob_val' +

                ' wHERE ( (~(~((~((~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) & m1conv.int_val))&(((~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val)) | m1conv.int_val)) & m2conv.int_val)) ' +
                ' & (~((~((~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) & m1conv.int_val))&(((~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val)) | m1conv.int_val)) | m2conv.int_val)) ' +
                '      = ?' +  # XOR between k and ptxt, mask1 and mask2
                '   AND tile_x = ? AND tile_y = ?' +
                '   AND trace_id IN (' +
                '       SELECT trace_id' +
                '       FROM traces' +
                '       WHERE tile_x = ? and tile_y = ?' +
                '       LIMIT ?)'
                , (sbox + 1, sbox + 1, mask1_pos, mask2_pos, sboxin, tile_x, tile_y, tile_x, tile_y, num_traces))
    return cur


# (~(a&b))&(a|b)

def select_traces_by_tile_two_sboxin_limitrange(db, tile_x, tile_y, sbox, sboxin, sbox2, sboxin2, num_traces):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific input value of 2 given s-boxes
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :param int sbox2: Number of the 2nd s-box (equals byte number)
    :param int sboxin2: Input value of the 2nd s-box (keybyte xor plaintext byte)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(k,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' JOIN byte_conv kconv' +
                '   ON substr(k,?,1) = kconv2.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv2.blob_val' +
                ' WHERE ((((~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) = ?)) AND ((~(kconv2.int_val&pconv2.int_val))&(kconv2.int_val|pconv2.int_val) = ?))))' +  # XOR between (k and ptxt) AND (k2 and ptxt2)
                '   AND tile_x = ? AND tile_y = ?' +
                '   AND trace_id IN (' +
                '       SELECT trace_id' +
                '       FROM traces' +
                '       WHERE tile_x = ? and tile_y = ?' +
                '       LIMIT ?)'
                ,
                (sbox + 1, sbox + 1, sbox2 + 1, sbox2 + 1, sboxin, sboxin2, tile_x, tile_y, tile_x, tile_y, num_traces))
    return cur


def select_traces_by_tile_sboxin_limit(db, tile_x, tile_y, sbox, sboxin, limit):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific input value at a given s-box
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(k,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' wHERE (~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) = ?' +  # XOR between k and ptxt
                '   AND tile_x = ? AND tile_y = ?' +
                ' LIMIT ?'
                , (sbox + 1, sbox + 1, sboxin, tile_x, tile_y, limit))
    return cur


def select_traces_by_tile_not_sboxin(db, tile_x, tile_y, sbox, sboxin):
    """
    Makes a query to the database and selects all traces at one tile with all but one
    specific input value at a given s-box
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(k,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' WHERE (~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) != ?' +  # XOR between k and ptxt
                '   AND tile_x = ? AND tile_y = ?'
                , (sbox + 1, sbox + 1, sboxin, tile_x, tile_y))
    return cur


def select_traces_by_tile_not_sboxin_limit(db, tile_x, tile_y, sbox, sboxin, limit):
    """
    Makes a query to the database and selects all traces at one tile with all but one
    specific input value at a given s-box
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int sboxin: Input value of the s-box (keybyte xor plaintext byte)
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(k,?,1) = kconv.blob_val' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' WHERE (~(kconv.int_val&pconv.int_val))&(kconv.int_val|pconv.int_val) != ?' +  # XOR between k and ptxt
                '   AND tile_x = ? AND tile_y = ?' +
                ' LIMIT ?'
                , (sbox + 1, sbox + 1, sboxin, tile_x, tile_y, limit))
    return cur


def select_traces_by_tile_arb_arg(db, tile_x, tile_y, **kwargs):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific input value at a given s-box
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param arg: name of arg to select
    :param val: value of arg
    :param pos: start of val to select in arg
    :param val_len: length of val
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    if 'arg' in kwargs:
        arbitrary_argument_to_select = kwargs['arg']
    else:
        _info.error("Please pass the select parameter by using arg='abitrary_param'")
        return None

    if 'val' in kwargs:
        arbitrary_value_to_select = int(kwargs['val'])
    else:
        _info.error("Please pass the value to the selected parameter by using val='abitrary_value'")
        return None

    if 'pos' in kwargs:
        arbitrary_value_position = int(kwargs['pos']) + 1
    else:
        _info.warning("No position specified, setting position to 1")
        arbitrary_value_position = 1

    if 'val_len' in kwargs:
        arbitrary_value_length = int(kwargs['val_len'])
    else:
        _info.warning("No length specified, setting length to 1 Byte")
        arbitrary_value_length = 1

    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(' + arbitrary_argument_to_select + ',?,?) = kconv.blob_val' +
                ' wHERE kconv.int_val = ?' +
                '   AND tile_x = ? AND tile_y = ?'
                , (arbitrary_value_position, arbitrary_value_length, arbitrary_value_to_select, tile_x, tile_y))
    return cur


def select_all_by_tile_arb_arg(db, tile_x, tile_y, **kwargs):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific input value at a given s-box
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param arg: name of arg to select
    :param val: value of arg
    :param pos: start of val to select in arg
    :param val_len: length of val
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    if 'arg' in kwargs:
        arbitrary_argument_to_select = kwargs['arg']
    else:
        _info.error("Please pass the select parameter by using arg='abitrary_param'")
        return None

    if 'val' in kwargs:
        arbitrary_value_to_select = int(kwargs['val'])
    else:
        _info.error("Please pass the value to the selected parameter by using val='abitrary_value'")
        return None

    if 'pos' in kwargs:
        arbitrary_value_position = int(kwargs['pos']) + 1
    else:
        _info.warning("No position specified, setting position to 1")
        arbitrary_value_position = 1

    if 'val_len' in kwargs:
        arbitrary_value_length = int(kwargs['val_len'])
    else:
        _info.warning("No length specified, setting length to 1 Byte")
        arbitrary_value_length = 1

    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(' + arbitrary_argument_to_select + ',?,?) = kconv.blob_val' +
                ' wHERE kconv.int_val = ?' +
                '   AND tile_x = ? AND tile_y = ?'
                , (arbitrary_value_position, arbitrary_value_length, arbitrary_value_to_select, tile_x, tile_y))
    return cur


def select_traces_by_tile_arb_arg_limitrange(db, tile_x, tile_y, limitrange, **kwargs):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific input value at a given s-box
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param arg: name of arg to select
    :param val: value of arg
    :param pos: start of val to select in arg
    :param val_len: length of val
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    if 'arg' in kwargs:
        arbitrary_argument_to_select = kwargs['arg']
    else:
        _info.error("Please pass the select parameter by using arg='abitrary_param'")
        return None

    if 'val' in kwargs:
        arbitrary_value_to_select = int(kwargs['val'])
    else:
        _info.error("Please pass the value to the selected parameter by using val='abitrary_value'")
        return None

    if 'pos' in kwargs:
        arbitrary_value_position = int(kwargs['pos']) + 1
    else:
        _info.warning("No position specified, setting position to 1")
        arbitrary_value_position = 1

    if 'val_len' in kwargs:
        arbitrary_value_length = int(kwargs['val_len'])
    else:
        _info.warning("No length specified, setting length to 1 Byte")
        arbitrary_value_length = 1

    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(' + arbitrary_argument_to_select + ',?,?) = kconv.blob_val' +
                ' wHERE kconv.int_val = ?' +
                '   AND tile_x = ? AND tile_y = ?' +
                '   AND trace_id IN (' +
                '   SELECT trace_id' +
                '   FROM traces' +
                '   WHERE tile_x = ? and tile_y = ?' +
                '   LIMIT ?)'
                , (arbitrary_value_position, arbitrary_value_length, arbitrary_value_to_select, tile_x, tile_y, tile_x,
                   tile_y, limitrange))
    return cur


def select_all_by_tile_arb_arg_limitrange(db, tile_x, tile_y, limitrange, **kwargs):
    """
    Makes a query to the database and selects all traces at one tile with a
    specific input value at a given s-box
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param arg: name of arg to select
    :param val: value of arg
    :param pos: start of val to select in arg
    :param val_len: length of val
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    if 'arg' in kwargs:
        arbitrary_argument_to_select = kwargs['arg']
    else:
        _info.error("Please pass the select parameter by using arg='abitrary_param'")
        return None

    if 'val' in kwargs:
        arbitrary_value_to_select = int(kwargs['val'])
    else:
        _info.error("Please pass the value to the selected parameter by using val='abitrary_value'")
        return None

    if 'pos' in kwargs:
        arbitrary_value_position = int(kwargs['pos'])
    else:
        _info.warning("No position specified, setting position to 1")
        arbitrary_value_position = 1

    if 'val_len' in kwargs:
        arbitrary_value_length = int(kwargs['val_len'])
    else:
        _info.warning("No length specified, setting length to 1 Byte")
        arbitrary_value_length = 1

    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv kconv' +
                '   ON substr(' + arbitrary_argument_to_select + ',?,?) = kconv.blob_val' +
                ' wHERE kconv.int_val = ?' +
                '   AND tile_x = ? AND tile_y = ?' +
                '   AND trace_id IN (' +
                '       SELECT trace_id' +
                '       FROM traces' +
                '       WHERE tile_x = ? and tile_y = ?' +
                '       LIMIT ?)'
                , (arbitrary_value_position, arbitrary_value_length, arbitrary_value_to_select, tile_x, tile_y, tile_x,
                   tile_y, limitrange))
    return cur


def select_traces_by_id(db, trace_ids):
    """
    Makes a query to the database and selects traces via trace_ids
    :param sqlite3.Connection db: open database connection
    :param int trace_ids: trace id or list of trace ids
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    if type(trace_ids) is int:
        trace_ids = [trace_ids]
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                'WHERE trace_id IN ({0})'.format(', '.join('?' for _ in trace_ids)), trace_ids)
    return cur


def get_trace_header_iterator(cur):
    """
    Generator function to fetch trace headers from cursor. A select call on the
    cursor must preceed this operation. Returns iterator over the traces yielding
    tuples.
    :param sqlite3.Cursor cur: cursor with available results
    :rtype list: trace header
    """
    row = cur.fetchone()
    while row != None:
        yield row
        row = cur.fetchone()


def get_trace_header_dict_iterator(cur):
    """
    Generator function to fetch trace headers from cursor as dictionary.
    :param sqlite3.Cursor cur: cursor with available results
    :rtype list: trace header
    """
    for hh in get_trace_header_iterator(cur):
        d = {kk: vv for kk, vv in zip(table_columns, hh)}
        d['k'] = blob_to_np_array(d['k'])
        d['iv'] = blob_to_np_array(d['iv'])
        d['ptxt'] = blob_to_np_array(d['ptxt'])
        d['ctxt'] = blob_to_np_array(d['ctxt'])
        d['adata'] = blob_to_np_array(d['adata'])
        yield d


def get_trace_header(cur):
    """
    Fetches one trace header from cursor. A select call on the cursor must
    preceed this operation. Returns tuple or None if no result is available.
    :param sqlite3.Cursor cur: cursor with available results
    :rtype numpy.array
    """
    row = cur.fetchone()
    if row == None:
        return None
    return row


def select_trace_header(db):
    """
    Makes a query to the database and selects all trace headers.
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[:-1])
    cur.execute('SELECT ' + columns +
                ' FROM traces')
    return cur


def select_trace_header_by_tile(db, tile_x, tile_y):
    """
    Makes a query to the database and selects all trace headers at one tile
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[:-1])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                'WHERE tile_x=? and tile_y=?', (tile_x, tile_y))
    return cur


def get_all_dict_iterator(cur):
    """
    Generator function to fetch trace headers from cursor as dictionary and the
    traces as numpy array and return them in a tuple.
    :param sqlite3.Cursor cur: cursor with available results
    :rtype tuple: trace header, traces
    """
    row = cur.fetchone()
    while row is not None:
        d = ({kk: vv for kk, vv in zip(table_columns, row[:-1])}, blob_to_np_array(row[-1:][0]))
        # convert all blobs to numpy arrays
        d[0]['k'] = blob_to_np_array(d[0]['k'])
        d[0]['iv'] = blob_to_np_array(d[0]['iv'])
        d[0]['ptxt'] = blob_to_np_array(d[0]['ptxt'])
        d[0]['ctxt'] = blob_to_np_array(d[0]['ctxt'])
        d[0]['adata'] = blob_to_np_array(d[0]['adata'])
        d[0]['start'] = d[0]['start']
        d[0]['end'] = d[0]['end']
        yield d
        row = cur.fetchone()


def select_all(db):
    """
    Makes a query to the database and selects all data.
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    cur.execute('SELECT *' +
                ' FROM traces')
    return cur


def select_all_by_tile(db, tile_x, tile_y):
    """
    Makes a query to the database and selects all data at one tile
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    cur.execute('SELECT *' +
                ' FROM traces ' +
                'WHERE tile_x=? and tile_y=?', (tile_x, tile_y))
    return cur


def select_all_by_tile_limit(db, tile_x, tile_y, offset, limit):
    """
    Makes a query to the database and selects all data at one tile
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    cur.execute('SELECT *' +
                ' FROM traces ' +
                'WHERE tile_x=? and tile_y=? LIMIT ? OFFSET ?', (tile_x, tile_y, limit, offset))
    return cur


def select_all_by_tile_sbox_traceid(db, tile_x, tile_y, sbox, start_id, steps):
    """
    Makes a query to the database and selects all traces from start_id
    to start_id + amount, at one tile with at a given byte position
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int start_id: starting point of the traces
    :param in steps: number of traces
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' WHERE trace_id >= ? AND trace_id < ?' +
                '   AND tile_x = ? AND tile_y = ?'
                , (sbox + 1, start_id, start_id + steps, tile_x, tile_y))
    return cur


def select_all_by_tile_sbox_without_traceid(db, tile_x, tile_y, sbox, start_id, steps):
    """
    Makes a query to the database and selects all traces except start_id
    to (start_id + amount), at one tile with a given byte position
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int start_id: starting point of the traces
    :param in steps: number of traces
    :param int start_id: starting point of the traces which are not included
    :param in steps: number of traces after the starting point which are not included
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                ' WHERE (trace_id < ? OR trace_id >= ?)' +
                '   AND tile_x = ? AND tile_y = ?'
                , (sbox + 1, start_id, start_id + steps, tile_x, tile_y))
    return cur


def select_all_by_tile_sbox_limit(db, tile_x, tile_y, sbox, offset, steps):
    """
    Makes a query to the database and selects all traces at one tile with a given byte position
    then selects "steps" many from starting row "offset"
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int offset: starting row of the traces
    :param in steps: number of traces
    :param int start_id: starting point of the traces which are not included
    :param in steps: number of traces after the starting point which are not included
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns)
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' JOIN byte_conv pconv' +
                '   ON substr(ptxt,?,1) = pconv.blob_val' +
                '   AND tile_x = ? AND tile_y = ?' +
                'LIMIT ? OFFSET ?'
                , (sbox + 1, tile_x, tile_y, steps, offset))
    return cur


def select_traces_where_adata_is_null(db, tile_x, tile_y):
    """
    Makes a query to the database and selects all traces at one tile with a given byte position
    then selects "steps" many from starting row "offset"
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int offset: starting row of the traces
    :param in steps: number of traces
    :param int start_id: starting point of the traces which are not included
    :param in steps: number of traces after the starting point which are not included
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' where adata is null' +
                '  AND tile_x = ? AND tile_y = ?'
                , (tile_x, tile_y))

    return cur


def select_traces_where_adata_is_null_limit(db, tile_x, tile_y, num_traces):
    """
    Makes a query to the database and selects all traces at one tile with a given byte position
    then selects "steps" many from starting row "offset"
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :param int sbox: Number of the s-box (equals byte number)
    :param int offset: starting row of the traces
    :param in steps: number of traces
    :param int start_id: starting point of the traces which are not included
    :param in steps: number of traces after the starting point which are not included
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    columns = ', '.join(table_columns[-1:])
    cur.execute('SELECT ' + columns +
                ' FROM traces ' +
                ' where adata is null' +
                '  AND tile_x = ? AND tile_y = ?' +
                '   AND trace_id IN (' +
                '   SELECT trace_id' +
                '   FROM traces' +
                '   WHERE tile_x = ? and tile_y = ?' +
                '   LIMIT ?)'
                , (tile_x, tile_y, tile_x, tile_y, num_traces))
    return cur


def select_max_of_column(db, column_name):
    """
    Makes a query to the database and selects all data.
    :param sqlite3.Connection db: open database connection
    :param int tile_x: X Coordinate of tile
    :param int tile_y: Y Coordinate of tile
    :rtype sqlite3.Cursor: Cursor with the query's result
    """
    cur = db.cursor()
    cur.execute('SELECT MAX(' + str(column_name) + ')' +
                ' FROM traces')
    return cur.fetchone()[0]
############################################################################################################

#######################################################################

# ***********************************************************************
# * Project: Attacking protected AES with Convolutional Neural Networks *
# *                                                                     *
# *                                                                     *
# * Purpose: It must run first to extract traces, plaintexts            *
# * and keys from database                                              *
# ***********************************************************************


def normalize(ndarr):
    """
    Takes an n-dimensions array and divides its element by the largest number within the array.
    :param ndarr: n-dimensions array.
    :returns n-dimensional array.
    """
    ndarr = ndarr.astype('float32')
    maxamp = np.ndarray.max(ndarr)
    ndarr /= maxamp
    return ndarr


def get_traces(db, cur, ids):
    """
    It extracts the traces from a cursor.
    :param sqlite3.Connection db: open database connection.
    :param sqlite3.Cursor cur   : cursor with desired results.
    :param int ids              : each trace has an id .
    :return n-dimensional array contains the traces that correspond to IDs.
    """
    # create a tensor to hold the traces: tensor [NrTraces][traceLength][1][1]
    trace_length = len(get_trace(select_traces_by_id(db, 1)))
    nr_traces = len(ids)
    traces = np.zeros(nr_traces * trace_length).reshape(nr_traces, trace_length)
    for i in range(nr_traces):
        traces[i] = get_trace(cur)
    traces = traces.reshape(nr_traces, trace_length, 1, 1)
    return traces


def get_keys(cur, nr_keys):
    """
    It stores the headers and their elements in a defined dictionary then extracts a specific number of
    keys from a cursor, it depends on the location of the keys in the database, they are located at col 7.
    :param sqlite3.Cursor cur: cursor with desired results.
    :param int nr_keys       : Number of desired keys.
    :retuns an array contains the keys.
    """
    i = 0
    h = {}
    for header in get_trace_header_iterator(cur):
        h[i] = header
        i = i + 1
        # keys are located in col 7
    key_length = len(blob_to_np_array(h[0][7]))
    keys = np.zeros(nr_keys * key_length).reshape(nr_keys, key_length)
    for i in range(nr_keys):
        keys[i] = blob_to_np_array(h[i][7])
    return keys


def get_plaintxt(cur, nr_plaintext):
    """
    It stores the headers and their elements in a defined dictionary then extracts a specific number of plain texts
    from a cursor, it dpends on the location of the  plain texts in the database, they are located at col 9.
    :param sqlite3.Cursor cur: cursor with desired results.
    :param nr_plaintext      : Number of desired plain texts.
    :retuns an array contains the plain texts.
    """
    i = 0
    h = {}
    for header in get_trace_header_iterator(cur):
        h[i] = header
        i = i + 1
        # plain texts are located in col 9
    ptxt_length = len(blob_to_np_array(h[0][9]))
    plaintexts = np.zeros(nr_plaintext * ptxt_length).reshape(nr_plaintext, ptxt_length)
    for i in range(nr_plaintext):
        plaintexts[i] = blob_to_np_array(h[i][9])
    return plaintexts


def get_ciphertxt(cur, nr_ctxt):
    """
    It stores the headers and their elements in a defined dictionary then extracts a specific number of cipher texts
    from a cursor, it dpends on the location of the  cipher texts in the database, they are located at col 10.
    :param sqlite3.Cursor cur: cursor with desired results.
    :param nr_ctxt           : Number of desired cipher texts.
    :retuns an array contains the cipher texts.
    """
    i = 0
    h = {}  # define a dictionary.
    for header in get_trace_header_iterator(cur):
        h[i] = header
        i = i + 1
    ctxt_length = len(blob_to_np_array(h[0][10]))
    ciphertext = np.zeros(nr_ctxt * ctxt_length).reshape(nr_ctxt, ctxt_length)
    for i in range(nr_ctxt):
        ciphertext[i] = blob_to_np_array(h[i][10])
    return ciphertext


def get_traces_by_sbox(cur):
    """
    Depends on the requested cursor it returns the corresponding traces.
    :para sqlite3.Cursor cur: cursor with desired results
    :return Number of found traces and n-dimensional array that holds those traces.
    """
    i = 0
    lst = []
    for row in get_trace_iterator(cur):
        i = i + 1
        lst.append(row)

    nr_traces = i
    trace_length = len(lst[0])
    traces = np.zeros(nr_traces * trace_length).reshape(nr_traces, trace_length)
    for j in range(nr_traces):
        traces[j] = lst[j]
    return i, traces


def plot_traces(nr_traces, x_traces):
    """
    It plot specified number of traces in one figure
    :param int nr_traces: desired number of traces
    :param float x_traces: n-dimensional array holds the traces
    """
    t = np.arange(0, len(x_traces[0]))
    for i in range(nr_traces):
        plt.figure(1)
        plt.subplot(nr_traces, 1, i + 1)
        plt.plot(t, x_traces[i].reshape(len(x_traces[0]), 1))
    plt.show()


# ***************************
# *     open database       *
# ***************************
db = open_db("Traces_Database/1x1x100000_r1_singlerail5_sr_hwpl.db")


# ***********************************************
# *     select traces id from 1 to 100000       *
# ***********************************************
IDs = range(1, 50001)
Nr_Traces = len(IDs)
cur = select_traces(db)
traces = get_traces(db, cur, IDs)
print('trace shape', traces.shape)


# ************************
# *     normalizing      *
# # ************************
for i in range(len(IDs)):
    traces[i] = normalize(traces[i])


# **********************************
# *  store traces in hdf5 format   *
# **********************************
f1 = h5py.File("Original_Traces/Original_Traces.hdf5", "w")
dSet1 = f1.create_dataset('O_T', data=traces)
f1.close()
print('*****************************************************\n')


# *************************************
# *     select the desired Keys       *
# *************************************
cur = select_trace_header_by_tile(db, 0, 0)
desiredKeys = Nr_Traces
keys = get_keys(cur, desiredKeys)
print('keys shape', keys.shape)
keyfile = h5py.File("Original_Keys/keys.hdf5", "w")
keySet = keyfile.create_dataset('O_keys', data=keys)
keyfile.close()
print('*******************************************************\n')


# ******************************************
# *     select the desired plaintext       *
# ******************************************
cur = select_trace_header_by_tile(db, 0, 0)
desiredPtxt = Nr_Traces
ptxts = get_plaintxt(cur, desiredPtxt)
print('plaintext shape', ptxts.shape)
ptxtfile = h5py.File("Original_Plaintexts/plaintxt.hdf5", "w")
ptxtSet = ptxtfile.create_dataset('P_txt', data=ptxts)
ptxtfile.close()


# **************************************
# *     Plot samples of traces         *
# **************************************
Number_Traces = 8
plot_traces(Number_Traces, traces)

# ***************************
# *    close database       *
# ***************************
close_db(db)

