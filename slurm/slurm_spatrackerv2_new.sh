#!/bin/bash

#==============================================================================
#                             CONFIGURAZIONE SLURM
#==============================================================================
#
# Nome del job.
#SBATCH --job-name=spatrackerv2_scratch
#
# File di output per i log.
#SBATCH --output=spatrackerv2_%j.log
#
# File di output per gli errori.
#SBATCH --error=spatrackerv2_%j.err
#
# Modalità di apertura per i file di log (append aggiunge in coda).
#SBATCH --open-mode=append
#
# Limite di tempo per il job.
#SBATCH --time=01:00:00
#
# Numero di task.
#SBATCH --ntasks=1
#
# Numero di core CPU per task.
#SBATCH --cpus-per-task=8
#
# Numero di GPU richieste.
#SBATCH --gpus=rtx_4090:2

#==============================================================================
#                SETUP E DEFINIZIONE DEI PERCORSI (MODIFICA QUI)
#==============================================================================

# Interrompe lo script immediatamente se un comando fallisce.
set -e 

echo "=== Job starting on $(hostname) at $(date) ==="

# --- 1. Definisci i percorsi permanenti su /work ---
# Cartella principale del progetto su /work
PROJECT_WORK_DIR="/cluster/work/igp_psr/niacobone/SpaTrackerV2"

# Cartella con i dati di input per questo specifico job
DATA_WORK_DIR="/cluster/work/igp_psr/niacobone/examples/test_recursive"

# Cartella dove verranno salvati i risultati finali (verrà creata se non esiste)
RESULTS_WORK_DIR="${PROJECT_WORK_DIR}/results"


# --- 2. Crea uno spazio di lavoro temporaneo sul disco veloce ---
# SLURM_SCRATCH è una variabile d'ambiente che punta al tuo disco scratch.
# Usiamo l'ID del job per creare una cartella unica.
JOB_SCRATCH_DIR="${SCRATCH}/spatracker_job_${SLURM_JOB_ID}"
echo "Creazione della directory di lavoro temporanea in: ${JOB_SCRATCH_DIR}"
mkdir -p "${JOB_SCRATCH_DIR}"

# --- 3. Copia le risorse necessarie su SCRATCH ---
echo "Copia del progetto e dei dati su SCRATCH..."
# Usiamo rsync per copiare la cartella del progetto (che contiene myenv2.tar.gz)
rsync -a --info=progress2 "${PROJECT_WORK_DIR}/" "${JOB_SCRATCH_DIR}/"

# Copiamo i dati di input in una sottocartella 'data' per mantenere l'ordine
rsync -a --info=progress2 "${DATA_WORK_DIR}/" "${JOB_SCRATCH_DIR}/data/"
echo "Copia completata."


#==============================================================================
#                     ESECUZIONE DEL JOB SU SCRATCH
#==============================================================================

# --- 4. Spostati nella directory di lavoro temporanea ---
cd "${JOB_SCRATCH_DIR}"
echo "Directory di lavoro attuale: $(pwd)"

# --- 5. Carica i moduli del cluster ---
module load stack/2024-06 python/3.11
echo "Moduli caricati: $(module list 2>&1)"

# --- 6. Prepara ed attiva l'ambiente virtuale ---
echo "Estrazione dell'ambiente virtuale (da myenv2.tar.gz)..."
# Questo è velocissimo perché avviene da SSD a SSD
tar -xzf myenv2.tar.gz
echo "Attivazione dell'ambiente virtuale..."
source myenv2/bin/activate
echo "Venv Python attivato: $(which python)"

# --- 7. Esegui lo script Python ---
# Nota come il percorso --data_dir ora è RELATIVO alla posizione su SCRATCH
echo "Avvio dell'inferenza di SpaTrackerV2..."
python inference.py --data_dir="./data"

echo "Inferenza terminata."
# È buona norma disattivare l'ambiente
deactivate


#==============================================================================
#                   SALVATAGGIO RISULTATI E PULIZIA
#==============================================================================

# --- 8. Copia i risultati importanti indietro su /work ---
# ASSUNZIONE: il tuo script salva gli output in una cartella chiamata 'output'
# Se il nome è diverso, modificalo qui.
# Creiamo una cartella unica per i risultati di questo job in /work.
FINAL_RESULTS_DIR="${RESULTS_WORK_DIR}/${SLURM_JOB_ID}"
echo "Copia dei risultati in: ${FINAL_RESULTS_DIR}"
mkdir -p "${FINAL_RESULTS_DIR}"
# Copiamo il contenuto della cartella di output da scratch alla destinazione permanente
rsync -a --info=progress2 ./output/ "${FINAL_RESULTS_DIR}/"

# --- 9. Pulisci la directory temporanea su SCRATCH ---
echo "Pulizia della directory temporanea su SCRATCH..."
rm -rf "${JOB_SCRATCH_DIR}"
echo "Pulizia completata."


#==============================================================================
#                                 CONCLUSIONE
#==============================================================================
echo "=== Job finished at $(date) ==="
start_time=${SLURM_JOB_START_TIME:-$(date +%s)}
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Total execution time: $(printf '%02d:%02d:%02d\n' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60))) (hh:mm:ss)"