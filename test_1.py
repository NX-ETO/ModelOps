from teradataml import *

eng = create_context(host = '192.168.0.37', username='dbc', password = 'dbc', database='DL_INICIAL')

qy_frac = '''
    SELECT BANDERA, cast(A_CONTEO as number)/cast(B_CONTEO as number) FRACCION
    FROM (
        SELECT *
        FROM (
            SELECT  BANDERA, GEST, COUNT(*) CONTEO
            FROM DL_INICIAL.NX_AUDIENCIAS_TRANSFORMADA_UNPV
            GROUP BY BANDERA, GEST
        ) A
        PIVOT ( SUM(CONTEO) AS CONTEO
                        FOR GEST IN (1 AS A, 0 AS B)
                        ) AS PVT
    ) A
'''
df = pd.read_sql(qy_frac, eng)

## ---
qy_del = '''
    delete from DL_INICIAL.NX_AUDIENCIAS_TRANSFORMADA_UNPV_BAL;
'''
pd.read_sql(qy_del, eng)
## ---

for i, f in df.iterrows():
    bandera = f['BANDERA']
    fraccion = f['FRACCION']

    query_0 = '''
        SELECT  *
        FROM DL_INICIAL.NX_AUDIENCIAS_TRANSFORMADA_UNPV
        WHERE GEST = 0
        AND BANDERA = '{}'
    '''.format(bandera)
    df_0 = DataFrame.from_query(query_0).sample(frac = round(fraccion, 2))

    copy_to_sql(df_0, table_name='NX_AUDIENCIAS_TRANSFORMADA_UNPV_BAL', if_exists='append')

query_1 = '''
    INSERT INTO DL_INICIAL.NX_AUDIENCIAS_TRANSFORMADA_UNPV_BAL
    SELECT A.*, 1 samp
    FROM DL_INICIAL.NX_AUDIENCIAS_TRANSFORMADA_UNPV A
    WHERE GEST = 1
'''
pd.read_sql(query_1, eng)


remove_context()