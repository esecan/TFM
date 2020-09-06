# -*- coding: utf-8 -*-

import psycopg2
from psycopg2.extensions import *

from utils.json_files import load_json

CONFIG = 'config/db.json'


class DBManager():
    '''Class that manages connections to the database.

    Keyword arguments:
        config_path (string) -- location of the JSON config file
    '''
    def __init__(self, config_path=CONFIG):
        self.params = "host='%(host)s' \
                       port='%(port)s' \
                       dbname='%(dbname)s' \
                       user='%(user)s' \
                       password='%(password)s'" % load_json(config_path)
        self.conn = None
        self.records = None


    def query(self, sql, fetch=False):
        '''Method for querying the database.

        Keyword arguments:
            sql (string) -- SQL query
            fetch (bool) -- in case the results need to be retrieved

        Output:
            records (list) -- (if fetch=True) results from the SQL
        '''
        self.records = None

        try:
            self.conn = psycopg2.connect(self.params)

            self.conn.autocommit = True

            # Creatings a new cursor
            cursor = self.conn.cursor()

            # Executing the SQL query
            cursor.execute(sql)

            if fetch:
                # Retrieving the records from the database
                self.records = cursor.fetchall()

            # Closing communication with the database
            cursor.close()

        except (Exception) as error:
            print('\n[ERROR\t\t]', error)
            print('[FETCH\t\t]', fetch)
            print('[QUERY\t\t]', sql, '\n')
            return False

        finally:
            if self.conn: self.conn.close()

        return self.records


    def check_user(self, username):
        '''Method for checking a user present in the database.

        Keyword arguments:
            values_dict (dict) -- username and password

        Output:
            records (list) -- if True, it is a valid combination
        '''
        sql = """SELECT username FROM users
                    WHERE username = lower('%s');""" \
                    % username

        return self.query(sql, fetch=True)


    def check_login(self, values_dict):
        '''Method for checking a user login info.

        Keyword arguments:
            values_dict (dict) -- username and password

        Output:
            records (list) -- if True, it is a valid combination
        '''
        sql = """SELECT username FROM users
                    WHERE username = lower('%(username)s')
                    AND password = crypt('%(password)s', password);""" \
                    % values_dict

        return self.query(sql, fetch=True)


    def create_user(self, values_dict):
        '''Method for creating a new user in the database.

        Keyword arguments:
            values_dict (dict) -- username and password

        Output:
            None -- creates the new entry in the database
        '''
        sql = """INSERT INTO users (username, password) VALUES
                    ('%(username)s',
                    crypt('%(password)s', gen_salt('bf', 8)));""" \
                    % values_dict

        self.query(sql)
