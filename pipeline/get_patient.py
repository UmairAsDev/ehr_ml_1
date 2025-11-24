import os
import sys
import pandas as pd
import numpy as np
import asyncio
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sqlalchemy import text
from sqlalchemy.orm import Session
from db.db import get_db
import json
from pandas import to_datetime



async def get_patient_ids(db: Session):
    try:
        """Fetch a single note and attach biopsy, general, mohs, and prescription results."""

        data = text("""SELECT * FROm progressNotes pn JOIN pnAssessment pa ON pn.noteId = pa.noteId WHERE pa.dxId IN (1120,
        1121,
        1122,
        1123,
        1124,
        1216,
        1596,
        1662,
        1663,
        1666,
        1667,
        1668,
        1830,
        1872,
        1895,
        2051,
        2052,
        2102,
        2256) AND pn.noteDate >= "2023-01-01 00:00:0000" """)
        dxd_result = db.execute(data)
        
        patient_ids = []

        for row in dxd_result.mappings():
            patient_ids.append(str(row['patientId']))
        
        return patient_ids
    except Exception as e:
        print(f"An error occurred: {e}")
        patient_ids = []
    return patient_ids


async def fetch_final_data(db: Session, patient_ids: list):
    """Fetch final data for given patient IDs."""
    try:
        final_df = """SELECT
            pn.noteId, pn.provider, pn.physician, pn.referringPhysician, pn.noteDate, pn.patientId,
            npn.complaints, npn.pastHistory, npn.assesment, npn.reviewofsystem, npn.currentmedication,
            npn.`procedure`, npn.biopsyNotes, npn.mohsNotes, npn.allergy, npn.examination, npn.patientSummary, npn.procedure, npn.assesment,
            group_concat(concat(dc.icd10Code, ' ', d.dxDescription)) AS diagnoses, pos.posName as PlaceOfService, CONCAT(p.firstName, ' ', p.lastName) as 'Rendering Provider', CONCAT(p2.firstName, ' ', p2.lastName) as 'Physician', CONCAT(p3.firstName, ' ', p3.lastName) as 'Referring Provider', CONCAT(p4.firstName, ' ', p4.lastName) as 'Billing Provider'
            FROM progressNotes pn
            LEFT JOIN providers p ON p.providerId = pn.provider
            LEFT JOIN providers p2 ON p2.providerId = pn.physician
            LEFT JOIN providers p3 ON p3.providerId = pn.referringPhysician
            LEFT JOIN providers p4 ON p4.providerId = pn.billingProvider
            LEFT JOIN newProgressNotes npn ON pn.noteId = npn.noteId
            LEFT JOIN placeOfService pos ON pos.posCodes = pn.placeOfService
            LEFT JOIN pnAssessment pa ON pa.noteId = pn.noteId
            LEFT JOIN diagnosis d ON d.dxId = pa.dxId
            LEFT JOIN diagnosisCodes dc ON dc.dxId = d.dxId AND dc.dxCodeId = pa.dxCodeId
            WHERE pn.physicianSignDate IS NOT NULL
            AND pn.patientId IN (""" + ",".join(patient_ids) + """) AND pn.noteDate >= "2023-01-01 00:00:0000" 
            GROUP BY pn.noteId"""
        final_result =db.execute(text(final_df)).fetchall()
        df = pd.DataFrame(final_result)
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        df = pd.DataFrame()
        return df


async def get_patient_data(db: Session, patient_ids: list):
    """Get patient data for given patient IDs."""
    try:
        patient_ids = await get_patient_ids(db)
        
        print(f"one patient id: {patient_ids[0]}")
        patient_id = patient_ids[0]
        
        df = await fetch_final_data(db, [patient_id])
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        df = pd.DataFrame()
        return df





if __name__ == '__main__':
    db = next(get_db())
    patient_data = asyncio.run(get_patient_data(db, ['178635']))
    print(patient_data.head())