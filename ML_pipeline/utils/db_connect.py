# -*- coding: utf-8 -*-

import pymysql
import psycopg2
from psycopg2.extensions import *


def connect(params=None, sql=None, tupla=None, fetch=False, platform='PostgreSQL'):
    """Auxiliary function for connecting to a database.

    Keyword arguments:
        params (string) -- host='', database='', user='', password=''
        sql (string) -- SQL query string
        tupla (tuple) -- variables to be formated into the SQL string
        fetch (bool) -- allows retrieving the records from the database
        platform (string) -- allows querying both PostgreSQL and MySQL

    Output:
        records (list) -- records obtained from querying the database
    """
    conn = records = None

    try:
        if   platform == 'PostgreSQL': conn = psycopg2.connect(params)
        elif platform == 'MySQL'     : conn = pymysql.connect(params)

        conn.autocommit = True

        # Creating a new cursor
        cursor = conn.cursor()

        # Executing the SQL query
        cursor.execute(sql)

        if fetch == True:
            # Retrieving the records from the database
            records = cursor.fetchall()

        # Closing communication with the database
        cursor.close()

    except (Exception) as error:
        print('\n[-] Error: ', error)
        print('[-] Fetch: ', fetch)
        print('[-] Query: ', sql, '\n')
        return False

    finally:
        if conn is not None:
            conn.close()

    return records
