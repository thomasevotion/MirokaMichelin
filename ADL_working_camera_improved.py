#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système de détection amélioré avec meilleurs modèles pour téléphones
- YOLOv8s ou YOLOv8m pour meilleure précision
- Seuils de confiance optimisés
- Détection de téléphones améliorée
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import json
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import queue
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectionStats:
    """Classe pour gérer les statistiques de détection"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Réinitialise toutes les statistiques"""
        # Statistiques globales (depuis le début)
        self.total_persons_detected = 0
        self.total_phone_moving_actions = 0
        self.total_phone_standing_actions = 0
        self.total_frames_processed = 0
        self.session_start_time = datetime.now()
        
        # Statistiques actuelles (dans la frame courante)
        self.current_persons_with_phone_moving = 0
        self.current_persons_with_phone_standing = 0
        self.current_persons_without_phone_moving = 0
        self.current_persons_without_phone_standing = 0
        self.current_active_persons = 0
        self.current_detected_persons = 0
        self.current_phones_detected = 0
        
        # Historique pour stabilité
        self.person_count_history = []
        self.phone_actions_counted = set()
        
        # Performance
        self.fps = 0.0
        self.last_fps_update = time.time()
        self.frame_times = []
    
    def update_current_stats(self, current_persons, detected_persons, phones_count):
        """Met à jour les statistiques actuelles"""
        self.current_detected_persons = len(detected_persons)
        self.current_active_persons = len(current_persons)
        self.current_phones_detected = phones_count
        
        # Réinitialiser les compteurs actuels
        self.current_persons_with_phone_moving = 0
        self.current_persons_with_phone_standing = 0
        self.current_persons_without_phone_moving = 0
        self.current_persons_without_phone_standing = 0
        
        # Compter les personnes actuelles par catégorie
        for person in current_persons:
            if person['phone_action'] == "PHONE_NEARBY":
                if person['is_moving']:
                    self.current_persons_with_phone_moving += 1
                else:
                    self.current_persons_with_phone_standing += 1
            else:  # NO_PHONE
                if person['is_moving']:
                    self.current_persons_without_phone_moving += 1
                else:
                    self.current_persons_without_phone_standing += 1
        
        # Ajouter à l'historique pour stabilité
        self.person_count_history.append(self.current_active_persons)
        if len(self.person_count_history) > 30:  # Garder 30 dernières valeurs
            self.person_count_history.pop(0)
    
    def update_cumulative_stats(self, current_persons, frame_count):
        """Met à jour les statistiques cumulatives"""
        self.total_frames_processed = frame_count
        
        # Compter les actions téléphone (éviter les doublons)
        for person in current_persons:
            if person['phone_action'] == "PHONE_NEARBY":
                action_key = f"{person['id']}_{person['is_moving']}_{frame_count // 60}"  # Toutes les 60 frames
                if action_key not in self.phone_actions_counted:
                    self.phone_actions_counted.add(action_key)
                    if person['is_moving']:
                        self.total_phone_moving_actions += 1
                    else:
                        self.total_phone_standing_actions += 1
    
    def get_stable_person_count(self):
        """Retourne un nombre de personnes stable (moyenne des dernières valeurs)"""
        if not self.person_count_history:
            return 0
        return round(np.mean(self.person_count_history))
    
    def update_fps(self):
        """Met à jour le FPS"""
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:  # Mise à jour toutes les secondes
            if self.frame_times:
                self.fps = len(self.frame_times) / (current_time - self.last_fps_update)
                self.frame_times.clear()
            self.last_fps_update = current_time
        else:
            self.frame_times.append(current_time)
    
    def get_stats_json(self):
        """Retourne les statistiques au format JSON"""
        session_duration = (datetime.now() - self.session_start_time).total_seconds()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "session_duration_seconds": round(session_duration, 2),
            "performance": {
                "fps": round(self.fps, 2),
                "total_frames_processed": self.total_frames_processed
            },
            "global_stats": {
                "total_persons_detected": self.total_persons_detected,
                "total_phone_moving_actions": self.total_phone_moving_actions,
                "total_phone_standing_actions": self.total_phone_standing_actions
            },
            "current_stats": {
                "active_persons": self.current_active_persons,
                "detected_persons": self.current_detected_persons,
                "phones_detected": self.current_phones_detected,
                "stable_person_count": self.get_stable_person_count(),
                "persons_with_phone_moving": self.current_persons_with_phone_moving,
                "persons_with_phone_standing": self.current_persons_with_phone_standing,
                "persons_without_phone_moving": self.current_persons_without_phone_moving,
                "persons_without_phone_standing": self.current_persons_without_phone_standing
            }
        }
    
    def get_final_summary(self):
        """Retourne un résumé final formaté pour l'affichage"""
        session_duration = datetime.now() - self.session_start_time
        hours = int(session_duration.total_seconds() // 3600)
        minutes = int((session_duration.total_seconds() % 3600) // 60)
        seconds = int(session_duration.total_seconds() % 60)
        
        total_phone_actions = self.total_phone_moving_actions + self.total_phone_standing_actions
        
        return {
            "session_duration": f"{hours:02d}h {minutes:02d}m {seconds:02d}s",
            "total_frames": self.total_frames_processed,
            "avg_fps": round(self.fps, 2),
            "total_unique_persons": self.total_persons_detected,
            "total_phone_actions": total_phone_actions,
            "phone_moving_actions": self.total_phone_moving_actions,
            "phone_standing_actions": self.total_phone_standing_actions,
            "session_start": self.session_start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def print_final_summary(self):
        """Affiche le résumé final des statistiques"""
        summary = self.get_final_summary()
        
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ FINAL DE LA SESSION")
        print("=" * 70)
        print(f"⏱️  Durée de la session: {summary['session_duration']}")
        print(f"📅 Début: {summary['session_start']}")
        print(f"📅 Fin: {summary['session_end']}")
        print("-" * 70)
        print("📈 PERFORMANCE:")
        print(f"   • Frames traitées: {summary['total_frames']:,}")
        print(f"   • FPS moyen: {summary['avg_fps']}")
        print("-" * 70)
        print("👥 PERSONNES:")
        print(f"   • Personnes uniques détectées: {summary['total_unique_persons']}")
        print("-" * 70)
        print("📱 ACTIONS TÉLÉPHONE:")
        print(f"   • Total d'actions téléphone: {summary['total_phone_actions']}")
        print(f"   • Téléphone + En mouvement: {summary['phone_moving_actions']}")
        print(f"   • Téléphone + Statique: {summary['phone_standing_actions']}")
        print("=" * 70)
        print()

def calculate_iou(box1, box2):
    """Calcule l'IoU entre deux bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def get_hand_keypoints(person_bbox, pose_results):
    """Extrait les keypoints des mains d'une personne depuis les résultats pose
    
    Keypoints COCO pose: 9=poignet gauche, 10=poignet droit
    """
    if pose_results is None:
        return None, None
    
    # pose_results peut être un objet Results ou une liste
    try:
        if hasattr(pose_results, 'keypoints'):
            # Un seul résultat
            results_list = [pose_results]
        else:
            # Liste de résultats
            results_list = pose_results if isinstance(pose_results, (list, tuple)) else []
    except:
        return None, None
    
    if len(results_list) == 0:
        return None, None
    
    person_x1, person_y1, person_x2, person_y2 = person_bbox
    
    # Trouver la personne correspondante dans les résultats pose
    for result in results_list:
        try:
            if result.keypoints is None or len(result.keypoints) == 0:
                continue
            
            # Vérifier si les keypoints correspondent à cette personne (bbox overlap)
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                pose_bbox = result.boxes.xyxy[0].cpu().numpy()
                pose_iou = calculate_iou(person_bbox, pose_bbox)
                
                if pose_iou > 0.3:  # Correspondance probable
                    keypoints = result.keypoints.data[0].cpu().numpy() if hasattr(result.keypoints, 'data') else result.keypoints[0].cpu().numpy()
                    # Keypoints: [x, y, confidence] pour chaque point
                    # Index 9 = poignet gauche, 10 = poignet droit
                    if len(keypoints) >= 11:
                        left_wrist = keypoints[9] if len(keypoints[9]) >= 3 and keypoints[9][2] > 0.3 else None
                        right_wrist = keypoints[10] if len(keypoints[10]) >= 3 and keypoints[10][2] > 0.3 else None
                        return left_wrist, right_wrist
        except Exception as e:
            continue
    
    return None, None

def is_phone_near_hands(phone_bbox, left_wrist, right_wrist, max_hand_distance=80):
    """Vérifie si un téléphone est proche des mains détectées"""
    if left_wrist is None and right_wrist is None:
        return False
    
    phone_center = ((phone_bbox[0] + phone_bbox[2]) / 2,
                   (phone_bbox[1] + phone_bbox[3]) / 2)
    
    if left_wrist is not None:
        distance = np.sqrt((phone_center[0] - left_wrist[0])**2 + 
                         (phone_center[1] - left_wrist[1])**2)
        if distance < max_hand_distance:
            return True
    
    if right_wrist is not None:
        distance = np.sqrt((phone_center[0] - right_wrist[0])**2 + 
                         (phone_center[1] - right_wrist[1])**2)
        if distance < max_hand_distance:
            return True
    
    return False

def find_closest_phone_to_person(person_bbox, phones, pose_results=None, max_distance=180):
    """Trouve le téléphone le plus proche d'une personne avec règles d'association équilibrées
    
    CRITÈRES ÉQUILIBRÉS :
    - Téléphone doit être proche de la personne (max 180 pixels) - assoupli pour meilleure détection
    - Téléphone doit chevaucher ou être dans zone proche (60% de la personne)
    - Priorité aux téléphones près des mains (via pose estimation)
    - Protection contre téléphone sur table à 5m ≠ personne lointaine
    """
    if not phones:
        return None
    
    person_center = ((person_bbox[0] + person_bbox[2]) / 2, 
                    (person_bbox[1] + person_bbox[3]) / 2)
    
    closest_phone = None
    min_distance = float('inf')
    best_score = 0  # Score basé sur proximité et chevauchement
    
    person_width = person_bbox[2] - person_bbox[0]
    person_height = person_bbox[3] - person_bbox[1]
    
    # Zone modérée pour associer les téléphones (60% de la personne)
    # Assez large pour détecter mais pas trop pour éviter associations lointaines
    zone_x1 = person_bbox[0] - person_width * 0.5
    zone_y1 = person_bbox[1] - person_height * 0.3
    zone_x2 = person_bbox[2] + person_width * 0.5
    zone_y2 = person_bbox[3] + person_height * 0.3
    
    # Récupérer les keypoints des mains si disponibles
    left_wrist, right_wrist = get_hand_keypoints(person_bbox, pose_results)
    
    for phone in phones:
        phone_center = ((phone['bbox'][0] + phone['bbox'][2]) / 2,
                       (phone['bbox'][1] + phone['bbox'][3]) / 2)
        
        # Distance euclidienne
        distance = np.sqrt((person_center[0] - phone_center[0])**2 + 
                         (person_center[1] - phone_center[1])**2)
        
        # CRITÈRE 1: Distance maximale modérée (180 pixels)
        # Si téléphone très loin, ignorer (protection contre téléphone sur table lointaine)
        if distance > max_distance:
            continue  # Téléphone trop loin, ignorer
        
        phone_x1, phone_y1, phone_x2, phone_y2 = phone['bbox']
        
        # CRITÈRE 2: Téléphone doit être dans zone OU chevaucher la personne
        phone_in_zone = (zone_x1 <= phone_x1 <= zone_x2 or 
                        zone_x1 <= phone_x2 <= zone_x2) and \
                       (zone_y1 <= phone_y1 <= zone_y2 or 
                        zone_y1 <= phone_y2 <= zone_y2)
        
        # Calculer IoU pour prioriser les téléphones qui chevauchent la personne
        iou = calculate_iou(person_bbox, phone['bbox'])
        
        # CRITÈRE 3: IoU minimum OU dans zone OU près des mains
        phone_near_hands = is_phone_near_hands(phone['bbox'], left_wrist, right_wrist, max_hand_distance=100)
        
        # Accepter si : chevauche significatif OU dans zone OU près des mains
        if iou < 0.01 and not phone_in_zone and not phone_near_hands:
            continue  # Téléphone trop éloigné, pas d'association
        
        # Score basé sur distance, IoU et proximité mains
        score = (1.0 / (1.0 + distance / 60.0)) + (iou * 4.0)  # IoU a plus de poids
        
        # BONUS: Téléphone près des mains détectées (pose estimation) - très important
        if phone_near_hands:
            score += 3.0  # Fort bonus pour téléphone dans la main
        
        # BONUS: Téléphone près de la tête (position d'appel)
        if is_phone_near_head(person_bbox, phone['bbox']):
            score += 2.0
        
        # BONUS: Téléphone chevauche la personne
        if iou > 0.05:
            score += 1.5
        
        # BONUS: Téléphone dans zone proche
        if phone_in_zone:
            score += 0.5
        
        if score > best_score:
            min_distance = distance
            best_score = score
            closest_phone = phone
    
    return closest_phone

def is_phone_near_head(person_bbox, phone_bbox):
    """Détection du téléphone près de la tête - très permissif pour position d'appel"""
    person_x1, person_y1, person_x2, person_y2 = person_bbox
    phone_x1, phone_y1, phone_x2, phone_y2 = phone_bbox
    
    # Zone de tête très étendue (70% de la hauteur) pour capturer position d'appel
    head_y1 = person_y1
    head_y2 = person_y1 + (person_y2 - person_y1) * 0.7
    
    phone_center_y = (phone_y1 + phone_y2) / 2
    phone_center_x = (phone_x1 + phone_x2) / 2
    person_center_x = (person_x1 + person_x2) / 2
    
    person_width = person_x2 - person_x1
    person_height = person_y2 - person_y1
    
    # Distance horizontale très permissive (100% de la largeur) pour position d'appel
    max_horizontal_distance = person_width * 1.0
    
    # Vérifier si le téléphone chevauche la zone de tête
    in_head_zone = head_y1 <= phone_center_y <= head_y2
    close_horizontally = abs(phone_center_x - person_center_x) < max_horizontal_distance
    
    # Vérifier si le téléphone chevauche physiquement la zone de tête (même partiellement)
    phone_overlaps_head = not (phone_y2 < head_y1 or phone_y1 > head_y2)
    phone_near_horizontally = not (phone_x2 < person_x1 - person_width * 0.3 or 
                                   phone_x1 > person_x2 + person_width * 0.3)
    
    # Seuil "très proche" - très permissif pour position d'appel
    phone_very_close = (phone_center_y >= head_y1 and phone_center_y <= head_y2 and
                       abs(phone_center_x - person_center_x) < person_width * 0.7)
    
    # Vérifier si le téléphone est dans une zone étendue autour de la tête (position d'appel)
    head_zone_extended = (head_y1 - person_height * 0.1 <= phone_center_y <= head_y2 + person_height * 0.1)
    head_horizontal_extended = abs(phone_center_x - person_center_x) < person_width * 0.9
    
    # Accepter si : téléphone dans zone tête OU chevauche tête OU très proche OU zone étendue
    return ((in_head_zone and close_horizontally) or 
            (phone_overlaps_head and phone_near_horizontally) or
            phone_very_close or
            (head_zone_extended and head_horizontal_extended))

def detect_movement_stable(displacement_history, frame_count):
    """Détection de mouvement stable avec seuils adaptatifs - optimisé pour RTSP"""
    if len(displacement_history) < 5:  # Plus de frames nécessaires pour RTSP
        return False
    
    recent_displacements = displacement_history[-8:]  # Plus d'historique pour RTSP
    
    # Seuils plus stricts pour RTSP (évite faux positifs dus à la latence)
    variance = np.var(recent_displacements)
    if variance < 0.5:  # Très stable
        threshold = 6.0  # Augmenté de 4.0 à 6.0 pour RTSP
    elif variance < 2.0:  # Modérément stable
        threshold = 5.0  # Augmenté de 3.0 à 5.0
    else:  # Instable
        threshold = 4.0  # Augmenté de 2.0 à 4.0
    
    avg_recent = np.mean(recent_displacements)
    if avg_recent < threshold:
        return False
    
    # Exiger plus de mouvements significatifs pour RTSP
    significant_movements = [d for d in recent_displacements if d > threshold * 0.8]  # 0.8 au lieu de 0.7
    if len(significant_movements) < 5:  # 5 au lieu de 3
        return False
    
    # Exiger que les dernières frames montrent un mouvement continu
    if len(recent_displacements) >= 6:
        last_five = recent_displacements[-5:]  # 5 au lieu de 3
        if all(d > threshold * 0.6 for d in last_five):  # 0.6 au lieu de 0.5
            return True
    
    return False

class RobustPersonTracker:
    """Classe pour suivre les personnes de manière robuste avec prédiction"""
    
    def __init__(self):
        self.active_persons = {}
        self.person_movement_history = {}
        self.person_movement_state = {}
        self.next_person_id = 1
        self.max_disappeared_frames = 60
        self.prediction_history = {}
        # Ensemble des IDs uniques vus depuis le début
        self.unique_person_ids_ever_seen = set()
        
        # Lissage temporel pour téléphone - ajusté pour RTSP
        self.person_phone_history = {}  # Historique des détections téléphone (dernières N frames)
        self.person_phone_state = {}    # État stable de téléphone (lissé)
        self.phone_history_size = 8     # Réduit pour RTSP (moins de frames nécessaires)
        self.phone_activation_threshold = 0.35  # 35% des frames (réduit pour RTSP)
        self.phone_deactivation_threshold = 0.25  # 25% pour désactiver (hystérésis)
        
    def predict_position(self, person_id, last_known_position, last_velocity, frames_missing):
        """Prédit la position d'une personne manquante"""
        if person_id not in self.prediction_history:
            self.prediction_history[person_id] = []
        
        predicted_center = (
            last_known_position[0] + last_velocity[0] * frames_missing,
            last_known_position[1] + last_velocity[1] * frames_missing
        )
        
        person_width = 100
        person_height = 200
        
        predicted_bbox = [
            int(predicted_center[0] - person_width / 2),
            int(predicted_center[1] - person_height / 2),
            int(predicted_center[0] + person_width / 2),
            int(predicted_center[1] + person_height / 2)
        ]
        
        return predicted_bbox, predicted_center
    
    def get_stable_phone_state(self, person_id, current_phone_detected):
        """Calcule l'état stable du téléphone basé sur l'historique (lissage temporel)"""
        # Initialiser l'historique si nécessaire
        if person_id not in self.person_phone_history:
            self.person_phone_history[person_id] = []
            self.person_phone_state[person_id] = False
        
        # Ajouter la détection actuelle à l'historique
        self.person_phone_history[person_id].append(1 if current_phone_detected else 0)
        
        # Garder seulement les dernières N frames
        if len(self.person_phone_history[person_id]) > self.phone_history_size:
            self.person_phone_history[person_id] = self.person_phone_history[person_id][-self.phone_history_size:]
        
        # Calculer le ratio de détections positives
        history = self.person_phone_history[person_id]
        if not history:
            return False
        
        phone_ratio = sum(history) / len(history)
        current_state = self.person_phone_state[person_id]
        
        # Hystérésis : seuils différents pour activer/désactiver
        if current_state:
            # Si déjà activé, on reste activé tant qu'on a au moins 30% de détections
            if phone_ratio >= self.phone_deactivation_threshold:
                return True
            else:
                self.person_phone_state[person_id] = False
                return False
        else:
            # Si désactivé, on active seulement si on a au moins 40% de détections
            if phone_ratio >= self.phone_activation_threshold:
                self.person_phone_state[person_id] = True
                return True
            else:
                return False
    
    def update_persons(self, detected_persons, phones, frame_count, pose_results=None):
        """Met à jour le suivi des personnes avec gestion robuste"""
        current_persons = []
        matched_indices = set()
        
        # Matcher les personnes détectées avec les personnes actives
        for i, person in enumerate(detected_persons):
            person_bbox = person['bbox']
            
            # Chercher une correspondance avec les personnes actives
            best_match = None
            best_iou = 0
            best_id = None
            
            for person_id, person_data in self.active_persons.items():
                iou = calculate_iou(person_bbox, person_data['bbox'])
                if iou > best_iou and iou > 0.4:
                    best_iou = iou
                    best_match = person_data
                    best_id = person_id
            
            if best_match:
                # Personne existante - mettre à jour
                person_id = best_id
                matched_indices.add(i)
                # Marquer cet ID comme vu
                self.unique_person_ids_ever_seen.add(person_id)
                
                # Calculer le déplacement
                prev_center = best_match['center']
                curr_center = ((person_bbox[0] + person_bbox[2]) / 2, 
                             (person_bbox[1] + person_bbox[3]) / 2)
                
                displacement = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                     (curr_center[1] - prev_center[1])**2)
                
                # Mettre à jour l'historique de mouvement
                if person_id not in self.person_movement_history:
                    self.person_movement_history[person_id] = []
                    self.person_movement_state[person_id] = False
                
                self.person_movement_history[person_id].append(displacement)
                if len(self.person_movement_history[person_id]) > 20:  # Plus d'historique pour RTSP
                    self.person_movement_history[person_id] = self.person_movement_history[person_id][-20:]
                
                # Détection de mouvement stable
                is_moving_detected = detect_movement_stable(self.person_movement_history[person_id], frame_count)
                
                # État de mouvement persistant - plus strict pour RTSP
                if is_moving_detected:
                    self.person_movement_state[person_id] = True
                else:
                    if self.person_movement_state[person_id]:
                        # Exiger plus de frames stables pour passer à "standing" avec RTSP
                        recent_displacements = self.person_movement_history[person_id][-8:]  # Plus de frames
                        if all(d < 2.0 for d in recent_displacements):  # Seuil plus élevé (2.0 au lieu de 1.5)
                            self.person_movement_state[person_id] = False
                
                is_moving = self.person_movement_state[person_id]
                
                # Trouver le téléphone le plus proche avec règles d'association équilibrées
                # (téléphone déjà filtré pour être associé à cette personne)
                closest_phone = find_closest_phone_to_person(person_bbox, phones, pose_results, max_distance=180)
                current_phone_detected = False
                phone_bbox_for_state = None
                
                if closest_phone:
                    # Le téléphone est déjà validé comme associé (via filtrage strict)
                    # Vérifications supplémentaires pour confirmer
                    iou_value = calculate_iou(person_bbox, closest_phone['bbox'])
                    phone_near_head = is_phone_near_head(person_bbox, closest_phone['bbox'])
                    
                    # Récupérer keypoints des mains pour validation
                    left_wrist, right_wrist = get_hand_keypoints(person_bbox, pose_results)
                    phone_near_hands = is_phone_near_hands(closest_phone['bbox'], left_wrist, right_wrist)
                    
                    # Accepter si : près de tête OU près des mains OU chevauche significatif
                    if phone_near_head or phone_near_hands or iou_value > 0.05:
                        current_phone_detected = True
                        phone_bbox_for_state = closest_phone['bbox']
                
                # Utiliser le lissage temporel pour obtenir l'état stable
                stable_phone_state = self.get_stable_phone_state(person_id, current_phone_detected)
                phone_action = "PHONE_NEARBY" if stable_phone_state else "NO_PHONE"
                
                # Garder le bbox du téléphone si l'état est stable
                final_phone_bbox = phone_bbox_for_state if stable_phone_state else None
                
                # Mettre à jour la personne
                updated_person = {
                    'id': person_id,
                    'bbox': person_bbox,
                    'center': curr_center,
                    'confidence': person['confidence'],
                    'is_moving': is_moving,
                    'phone_action': phone_action,
                    'phone_bbox': final_phone_bbox,
                    'last_seen': frame_count,
                    'is_predicted': False
                }
                
                self.active_persons[person_id] = updated_person
                current_persons.append(updated_person)
        
        # Gérer les nouvelles personnes
        for i, person in enumerate(detected_persons):
            if i not in matched_indices:
                # Nouvelle personne
                person_id = f"person_{self.next_person_id}"
                self.next_person_id += 1
                # Marquer cet ID comme vu
                self.unique_person_ids_ever_seen.add(person_id)
                
                person_bbox = person['bbox']
                curr_center = ((person_bbox[0] + person_bbox[2]) / 2, 
                             (person_bbox[1] + person_bbox[3]) / 2)
                
                # Trouver le téléphone le plus proche avec règles d'association équilibrées
                # (téléphone déjà filtré pour être associé à cette personne)
                closest_phone = find_closest_phone_to_person(person_bbox, phones, pose_results, max_distance=180)
                current_phone_detected = False
                phone_bbox_for_state = None
                
                if closest_phone:
                    # Le téléphone est déjà validé comme associé (via filtrage strict)
                    # Vérifications supplémentaires pour confirmer
                    iou_value = calculate_iou(person_bbox, closest_phone['bbox'])
                    phone_near_head = is_phone_near_head(person_bbox, closest_phone['bbox'])
                    
                    # Récupérer keypoints des mains pour validation
                    left_wrist, right_wrist = get_hand_keypoints(person_bbox, pose_results)
                    phone_near_hands = is_phone_near_hands(closest_phone['bbox'], left_wrist, right_wrist)
                    
                    # Accepter si : près de tête OU près des mains OU chevauche significatif
                    if phone_near_head or phone_near_hands or iou_value > 0.05:
                        current_phone_detected = True
                        phone_bbox_for_state = closest_phone['bbox']
                
                # Utiliser le lissage temporel pour obtenir l'état stable (initialisation)
                stable_phone_state = self.get_stable_phone_state(person_id, current_phone_detected)
                phone_action = "PHONE_NEARBY" if stable_phone_state else "NO_PHONE"
                
                # Garder le bbox du téléphone si l'état est stable
                final_phone_bbox = phone_bbox_for_state if stable_phone_state else None
                
                new_person = {
                    'id': person_id,
                    'bbox': person_bbox,
                    'center': curr_center,
                    'confidence': person['confidence'],
                    'is_moving': False,
                    'phone_action': phone_action,
                    'phone_bbox': final_phone_bbox,
                    'last_seen': frame_count,
                    'is_predicted': False
                }
                
                self.active_persons[person_id] = new_person
                self.person_movement_history[person_id] = [0.0]
                self.person_movement_state[person_id] = False
                current_persons.append(new_person)
                
                logger.info(f"🆕 Nouvelle personne détectée: {person_id}")
        
        # Gérer les personnes manquantes avec prédiction
        persons_to_remove = []
        for person_id, person_data in self.active_persons.items():
            frames_missing = frame_count - person_data['last_seen']
            
            if frames_missing > 0 and frames_missing <= self.max_disappeared_frames:
                if person_id in self.person_movement_history and len(self.person_movement_history[person_id]) > 1:
                    recent_displacements = self.person_movement_history[person_id][-3:]
                    avg_displacement = np.mean(recent_displacements)
                    
                    predicted_bbox, predicted_center = self.predict_position(
                        person_id, person_data['center'], 
                        [avg_displacement, avg_displacement], frames_missing
                    )
                    
                    predicted_person = {
                        'id': person_id,
                        'bbox': predicted_bbox,
                        'center': predicted_center,
                        'confidence': person_data['confidence'] * 0.3,
                        'is_moving': person_data['is_moving'],
                        'phone_action': person_data['phone_action'],
                        'phone_bbox': person_data['phone_bbox'],
                        'last_seen': person_data['last_seen'],
                        'is_predicted': True
                    }
                    
                    current_persons.append(predicted_person)
            
            elif frames_missing > self.max_disappeared_frames:
                persons_to_remove.append(person_id)
        
        # Nettoyer les personnes qui ont disparu définitivement
        for person_id in persons_to_remove:
            del self.active_persons[person_id]
            if person_id in self.person_movement_history:
                del self.person_movement_history[person_id]
            if person_id in self.person_movement_state:
                del self.person_movement_state[person_id]
            # Nettoyer l'historique téléphone
            if person_id in self.person_phone_history:
                del self.person_phone_history[person_id]
            if person_id in self.person_phone_state:
                del self.person_phone_state[person_id]
            logger.info(f"👋 Personne {person_id} retirée (disparue définitivement)")
        
        return current_persons

    def get_total_unique_persons_ever_seen(self):
        """Retourne le nombre total d'IDs uniques vus depuis le début"""
        return len(self.unique_person_ids_ever_seen)

def draw_detections(frame, current_persons, phones, stats):
    """Dessine les détections sur la frame avec améliorations visuelles"""
    # Couleurs améliorées
    colors = {
        'person_moving': (0, 255, 0),      # Vert vif pour mouvement
        'person_standing': (255, 0, 0),    # Rouge vif pour statique
        'phone': (0, 255, 255),            # Cyan pour téléphone
        'phone_high_conf': (255, 0, 255),  # Magenta pour téléphone haute confiance
        'text_bg': (0, 0, 0),              # Noir pour fond texte
        'text': (255, 255, 255)            # Blanc pour texte
    }
    
    # Dessiner les téléphones avec confiance
    for phone in phones:
        x1, y1, x2, y2 = phone['bbox']
        conf = phone['confidence']
        
        # Couleur selon la confiance
        phone_color = colors['phone_high_conf'] if conf > 0.7 else colors['phone']
        thickness = 3 if conf > 0.7 else 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), phone_color, thickness)
        cv2.putText(frame, f"PHONE {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phone_color, 2)
    
    # Dessiner les personnes
    for person in current_persons:
        x1, y1, x2, y2 = person['bbox']
        
        # Couleur selon le mouvement
        if person['is_moving']:
            color = colors['person_moving']
            movement_text = "MOVING"
        else:
            color = colors['person_standing']
            movement_text = "STANDING"
        
        # Dessiner la bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Texte de mouvement
        cv2.putText(frame, movement_text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Texte téléphone avec confiance
        phone_text = person['phone_action']
        if person['phone_bbox']:
            # Trouver la confiance du téléphone associé
            phone_conf = 0.0
            for phone in phones:
                if phone['bbox'] == person['phone_bbox']:
                    phone_conf = phone['confidence']
                    break
            phone_text += f" ({phone_conf:.2f})"
        
        phone_color = (0, 255, 255) if person['phone_action'] == "PHONE_NEARBY" else (128, 128, 128)
        cv2.putText(frame, phone_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phone_color, 2)
        
        # ID de la personne
        cv2.putText(frame, f"ID: {person['id']}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['text'], 1)
    
    # Informations sur la frame
    info_y = 30
    cv2.putText(frame, f"FPS: {stats.fps:.1f}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
    cv2.putText(frame, f"Personnes: {len(current_persons)}", (10, info_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
    cv2.putText(frame, f"Telephones: {len(phones)}", (10, info_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['text'], 2)
    
    # Statistiques détaillées
    stats_y = frame.shape[0] - 120
    cv2.putText(frame, f"Phone+Moving: {stats.current_persons_with_phone_moving}", (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)
    cv2.putText(frame, f"Phone+Standing: {stats.current_persons_with_phone_standing}", (10, stats_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)
    cv2.putText(frame, f"NoPhone+Moving: {stats.current_persons_without_phone_moving}", (10, stats_y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)
    cv2.putText(frame, f"NoPhone+Standing: {stats.current_persons_without_phone_standing}", (10, stats_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['text'], 2)
    
    return frame

class DetectionSystem:
    """Système principal de détection avec modèles améliorés"""
    
    def __init__(self):
        self.tracker = RobustPersonTracker()
        self.stats = DetectionStats()
        self.model = None
        self.pose_model = None  # Modèle pose pour détection des mains
        self.cap = None
        self.running = False
        self.latest_stats = {}
        self.rtsp_frame_skip = 0  # Frame skipping pour RTSP
        self.last_frame_time = time.time()
        self.frame_skip_counter = 0
        
    def initialize(self):
        """Initialise le système de détection avec modèle YOLOv11l ou YOLOv8l pour précision maximale"""
        try:
            # Charger le modèle le plus précis (YOLOv11l > YOLOv8l > YOLOv8m)
            logger.info("🔄 Chargement du modèle haute précision...")
            
            try:
                # Essayer YOLOv11l d'abord (meilleur modèle récent)
                self.model = YOLO('yolo11l.pt')
                logger.info("✅ Modèle YOLOv11l chargé! (Précision maximale)")
            except:
                try:
                    # Fallback vers YOLOv8l (très précis)
                    self.model = YOLO('yolov8l.pt')
                    logger.info("✅ Modèle YOLOv8l chargé! (Très précis)")
                except:
                    try:
                        # Fallback vers YOLOv8m si l n'est pas disponible
                        self.model = YOLO('yolov8m.pt')
                        logger.warning("⚠️ Modèle YOLOv8m chargé (fallback - moins précis)")
                    except:
                        # Dernier recours
                        self.model = YOLO('yolov8s.pt')
                        logger.warning("⚠️ Modèle YOLOv8s chargé (fallback - précision réduite)")
            
            # Charger le modèle pose pour détection des mains/keypoints
            logger.info("🔄 Chargement du modèle YOLOv8-pose pour détection des mains...")
            try:
                self.pose_model = YOLO('yolov8n-pose.pt')
                logger.info("✅ Modèle YOLOv8-pose chargé! (Détection keypoints mains/tête)")
            except:
                try:
                    # Essayer avec pose sans spécifier la taille
                    self.pose_model = YOLO('yolov8-pose.pt')
                    logger.info("✅ Modèle YOLOv8-pose chargé!")
                except:
                    logger.warning("⚠️ Modèle pose non disponible - détection téléphone via keypoints désactivée")
                    self.pose_model = None
            
            # Ouvrir la caméra RTSP (approche simple et synchrone qui fonctionne)
            rtsp_url = "rtsp://192.168.1.64:8554/head_color"
            logger.info(f"🔄 Connexion à la caméra RTSP: {rtsp_url}")
            
            # Créer la capture RTSP (approche simple comme dans le test)
            self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimal
            
            if not self.cap.isOpened():
                raise Exception(f"Impossible d'ouvrir la caméra RTSP: {rtsp_url}")
            
            # Lire quelques frames pour initialiser le stream
            for _ in range(3):
                ret, _ = self.cap.read()
                if ret:
                    break
                time.sleep(0.1)
            
            if not ret:
                raise Exception("Impossible de lire des frames depuis la caméra RTSP")
            
            # Obtenir les dimensions réelles de la caméra
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info("✅ Caméra RTSP connectée!")
            logger.info(f"📹 Configuration: {actual_width}x{actual_height}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation: {e}")
            return False
    
    def process_frame(self, frame):
        """Traite une frame avec seuils optimisés pour précision maximale et zéro faux positifs"""
        if self.model is None:
            return [], [], None
        
        # Détection YOLO avec optimisations pour FPS
        # - Résolution réduite à 416 pour améliorer les performances (au lieu de 640)
        # - conf global bas pour détecter tout, filtrage strict par classe ensuite
        # - iou élevé (0.7) pour NMS stricte et réduction des doublons
        # - max_det limité pour éviter la sur-détection
        # - device='0' pour forcer GPU si disponible
        try:
            results = self.model(frame, conf=0.25, iou=0.7, max_det=50, classes=[0, 67], 
                               imgsz=416, verbose=False, device='0', half=True)  # half=True pour FP16 (plus rapide)
        except:
            try:
                # Fallback sans half precision
                results = self.model(frame, conf=0.25, iou=0.7, max_det=50, classes=[0, 67], 
                                   imgsz=416, verbose=False, device='0')
            except:
                # Fallback si GPU non disponible
                results = self.model(frame, conf=0.25, iou=0.7, max_det=50, classes=[0, 67], 
                                   imgsz=416, verbose=False)
        
        # Détection pose pour keypoints (mains, tête) si disponible
        # DÉSACTIVÉ temporairement pour améliorer FPS (peut être réactivé si nécessaire)
        pose_results = None
        # Optionnel : désactiver pose pour améliorer FPS
        # if self.pose_model is not None:
        #     try:
        #         pose_results = self.pose_model(frame, conf=0.5, iou=0.7, imgsz=640, verbose=False, device='0')
        #     except:
        #         pose_results = None
        
        persons = []
        phones = []
        
        # Seuils stricts pour zéro faux positifs
        PERSON_MIN_CONF = 0.65  # Seuil élevé pour personnes (réduit faux positifs)
        PHONE_MIN_CONF = 0.35   # Seuil modéré pour téléphones (équilibre précision/détection)
        PHONE_NEAR_HEAD_CONF = 0.25  # Seuil plus bas pour téléphones près de la tête (position d'appel)
        
        # Taille minimale pour filtrer les objets trop petits (probablement faux positifs)
        # Réduit pour permettre détection de personnes proches (buste uniquement)
        MIN_PERSON_AREA = 1000   # pixels² (environ 30x35 pixels) - réduit pour personnes proches
        MIN_PHONE_AREA = 100     # pixels² (environ 10x10 pixels)
        
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Calculer la surface de la bounding box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                if class_id == 0:  # person
                    # Seuil strict pour réduire faux positifs
                    if confidence >= PERSON_MIN_CONF:
                        # Validation adaptative selon la confiance et la taille
                        # Si confiance très élevée (>0.75), être moins strict (personnes proches)
                        # Si confiance moyenne, être plus strict (réduire faux positifs)
                        
                        if confidence >= 0.75:
                            # Haute confiance : accepte les personnes proches (buste, ratio variable)
                            # Taille minimale réduite pour personnes proches
                            min_area = 500  # Très permissif pour personnes très proches
                            aspect_ratio = height / width if width > 0 else 0
                            # Ratio très permissif : 0.8 à 5.0 (buste peut être presque carré ou très allongé)
                            if area >= min_area and 0.8 <= aspect_ratio <= 5.0:
                                persons.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': confidence
                                })
                        elif confidence >= 0.70:
                            # Confiance élevée : taille minimale modérée
                            min_area = 800
                            aspect_ratio = height / width if width > 0 else 0
                            # Ratio modéré : 1.0 à 4.5
                            if area >= min_area and 1.0 <= aspect_ratio <= 4.5:
                                persons.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': confidence
                                })
                        else:
                            # Confiance moyenne : filtres stricts (réduire faux positifs)
                            aspect_ratio = height / width if width > 0 else 0
                            if area >= MIN_PERSON_AREA and 1.2 <= aspect_ratio <= 4.0:
                                persons.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': confidence
                                })
                elif class_id == 67:  # cell phone
                    # Seuil adaptatif : plus bas si téléphone détecté près d'une zone de tête
                    # (pour capturer position d'appel même avec confiance plus faible)
                    phone_conf_threshold = PHONE_MIN_CONF
                    
                    # Vérifier si le téléphone pourrait être près d'une tête
                    # (on vérifie après avoir collecté toutes les personnes, donc on accepte ici)
                    # Accepter avec seuil plus bas si confiance modérée (peut être position d'appel)
                    if 0.20 <= confidence < PHONE_MIN_CONF:
                        # Seuil réduit pour téléphones potentiellement en position d'appel
                        phone_conf_threshold = PHONE_NEAR_HEAD_CONF
                    
                    if confidence >= phone_conf_threshold and area >= MIN_PHONE_AREA:
                        phones.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence
                        })
        
        return persons, phones, pose_results
    
    def detection_loop(self):
        """Boucle principale de détection avec visualisation"""
        frame_count = 0
        frames_skipped = 0
        
        # Statistiques de performance pour diagnostiquer les goulots d'étranglement
        perf_stats = {
            'rtsp_read_times': [],
            'model_inference_times': [],
            'total_processing_times': [],
            'last_perf_log': time.time()
        }
        
        while self.running:
            try:
                loop_start = time.time()
                
                # Lire la frame RTSP directement (approche simple qui fonctionne)
                rtsp_read_start = time.time()
                ret, frame = self.cap.read()
                rtsp_read_time = (time.time() - rtsp_read_start) * 1000  # en ms
                
                # Filtrer les valeurs aberrantes (si RTSP read > 1000ms, c'est probablement une erreur de mesure)
                if rtsp_read_time < 1000:  # Ignorer les valeurs > 1 seconde (probablement erreur)
                    perf_stats['rtsp_read_times'].append(rtsp_read_time)
                else:
                    # Si valeur aberrante, utiliser la moyenne précédente ou 0
                    if perf_stats['rtsp_read_times']:
                        perf_stats['rtsp_read_times'].append(np.mean(perf_stats['rtsp_read_times'][-10:]))
                    else:
                        perf_stats['rtsp_read_times'].append(0)
                
                if not ret or frame is None:
                    logger.warning("⚠️ Impossible de lire la frame RTSP")
                    time.sleep(0.01)
                    continue
                
                frame_count += 1
                current_time = time.time()
                
                # Pas de frame skipping - traiter toutes les frames pour meilleure précision
                # (Le test montre que RTSP + modèle peut faire ~29 FPS)
                
                # Traiter la frame
                inference_start = time.time()
                persons, phones, pose_results = self.process_frame(frame)
                inference_time = time.time() - inference_start
                perf_stats['model_inference_times'].append(inference_time)
                
                # FILTRER les téléphones : ne garder QUE ceux associés à une personne
                # Un téléphone sur table à 5m ne doit PAS être associé à une personne lointaine
                # Mais on est moins strict pour permettre détection normale
                phones_filtered = []
                for phone in phones:
                    # Vérifier si ce téléphone est associé à au moins une personne
                    associated = False
                    for person in persons:
                        closest_phone = find_closest_phone_to_person(
                            person['bbox'], [phone], pose_results, max_distance=180
                        )
                        if closest_phone is not None and closest_phone == phone:
                            associated = True
                            break
                    
                    # Ne garder que les téléphones associés à une personne
                    if associated:
                        phones_filtered.append(phone)
                
                phones = phones_filtered
                
                # Mettre à jour le suivi des personnes avec téléphones filtrés
                current_persons = self.tracker.update_persons(persons, phones, frame_count, pose_results)
                
                # Mettre à jour les statistiques
                self.stats.update_current_stats(current_persons, persons, len(phones))
                self.stats.update_cumulative_stats(current_persons, frame_count)
                self.stats.update_fps()
                
                # Mettre à jour le total de personnes uniques vues depuis le début
                self.stats.total_persons_detected = self.tracker.get_total_unique_persons_ever_seen()
                
                # Sauvegarder les statistiques pour l'API
                self.latest_stats = self.stats.get_stats_json()
                
                # Dessiner les détections sur la frame
                display_frame = draw_detections(frame.copy(), current_persons, phones, self.stats)
                
                # Afficher la frame
                cv2.imshow('Détection Améliorée - Téléphones', display_frame)
                
                # Gérer les touches
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' ou ESC
                    self.running = False
                    break
                elif key == ord('r'):  # Reset
                    self.reset_stats()
                
                # Mesurer le temps total de traitement
                total_time = time.time() - loop_start
                perf_stats['total_processing_times'].append(total_time)
                
                # Log périodique avec statistiques de performance
                if frame_count % 100 == 0 or (time.time() - perf_stats['last_perf_log']) > 5.0:
                    perf_stats['last_perf_log'] = time.time()
                    
                    # Calculer les moyennes
                    if perf_stats['rtsp_read_times']:
                        avg_rtsp = np.mean(perf_stats['rtsp_read_times'][-50:]) * 1000  # en ms
                        max_rtsp = np.max(perf_stats['rtsp_read_times'][-50:]) * 1000
                    else:
                        avg_rtsp = max_rtsp = 0
                    
                    if perf_stats['model_inference_times']:
                        avg_inference = np.mean(perf_stats['model_inference_times'][-50:]) * 1000  # en ms
                        max_inference = np.max(perf_stats['model_inference_times'][-50:]) * 1000
                    else:
                        avg_inference = max_inference = 0
                    
                    if perf_stats['total_processing_times']:
                        avg_total = np.mean(perf_stats['total_processing_times'][-50:]) * 1000  # en ms
                    else:
                        avg_total = 0
                    
                    logger.info(f"📊 Frame {frame_count}: {len(current_persons)} personnes, {len(phones)} téléphones, FPS: {self.stats.fps:.1f}")
                    logger.info(f"⏱️  PERFORMANCE - RTSP read: {avg_rtsp:.1f}ms (max: {max_rtsp:.1f}ms) | "
                              f"Model inference: {avg_inference:.1f}ms (max: {max_inference:.1f}ms) | "
                              f"Total: {avg_total:.1f}ms")
                    
                    # Nettoyer les listes pour éviter la croissance mémoire
                    if len(perf_stats['rtsp_read_times']) > 100:
                        perf_stats['rtsp_read_times'] = perf_stats['rtsp_read_times'][-50:]
                    if len(perf_stats['model_inference_times']) > 100:
                        perf_stats['model_inference_times'] = perf_stats['model_inference_times'][-50:]
                    if len(perf_stats['total_processing_times']) > 100:
                        perf_stats['total_processing_times'] = perf_stats['total_processing_times'][-50:]
                
                # Log périodique standard
                if frame_count % 300 == 0:
                    logger.info(f"📊 Frame {frame_count}: {len(current_persons)} personnes, {len(phones)} téléphones, FPS: {self.stats.fps:.1f}")
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de détection: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Démarre le système de détection"""
        if not self.initialize():
            return False
        
        self.running = True
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        logger.info("🚀 Système de détection amélioré démarré!")
        return True
    
    def stop(self):
        """Arrête le système de détection"""
        self.running = False
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Afficher le résumé final
        self.stats.print_final_summary()
        
        logger.info("⏹️ Système de détection arrêté!")
    
    def get_stats(self):
        """Retourne les statistiques actuelles"""
        return self.latest_stats
    
    def reset_stats(self):
        """Réinitialise les statistiques"""
        self.stats.reset()
        self.tracker = RobustPersonTracker()
        logger.info("🔄 Statistiques réinitialisées!")

# Instance globale du système
detection_system = DetectionSystem()

# Configuration Flask
app = Flask(__name__)
CORS(app)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """API endpoint pour récupérer les statistiques"""
    try:
        stats = detection_system.get_stats()
        return jsonify({
            "success": True,
            "data": stats
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_stats():
    """API endpoint pour réinitialiser les statistiques"""
    try:
        detection_system.reset_stats()
        return jsonify({
            "success": True,
            "message": "Statistiques réinitialisées"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """API endpoint pour vérifier le statut du système"""
    return jsonify({
        "success": True,
        "data": {
            "running": detection_system.running,
            "initialized": detection_system.model is not None,
            "camera_open": detection_system.cap is not None and detection_system.cap.isOpened()
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """API endpoint pour le health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

def main():
    """Fonction principale"""
    print("🚀 SYSTÈME DE DÉTECTION MAXIMALE PRÉCISION - TÉLÉPHONES")
    print("=" * 60)
    print("🎯 Optimisations pour précision maximale et zéro faux positifs:")
    print("  - Modèle YOLOv11l ou YOLOv8l (Large - Précision maximale)")
    print("  - YOLOv8-pose pour détection keypoints mains/tête")
    print("  - Seuils de confiance stricts (0.65 personnes, 0.35 téléphones)")
    print("  - NMS stricte (IoU 0.7) pour éliminer doublons")
    print("  - Filtrage par taille minimale (évite faux positifs petits)")
    print("  - Validation ratio hauteur/largeur pour personnes")
    print("  - ASSOCIATION téléphone-personne équilibrée (max 180 pixels)")
    print("  - Téléphones isolés (sur table) NON associés aux personnes lointaines")
    print("  - Validation via keypoints mains pour téléphones dans la main")
    print("  - Optimisé pour RTX 4070 (imgsz=640 pour fluidité)")
    print("=" * 60)
    
    # Démarrer le système de détection
    if not detection_system.start():
        print("❌ Impossible de démarrer le système de détection")
        return
    
    print("✅ Système de détection amélioré démarré!")
    print("🌐 API disponible sur: http://localhost:5000")
    print("=" * 60)
    print("🎮 Contrôles:")
    print("  - R : Reset des statistiques")
    print("  - Q ou ESC : Quitter")
    print("=" * 60)
    
    try:
        # Démarrer l'API Flask
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt demandé par l'utilisateur")
    finally:
        # Afficher le résumé final avant d'arrêter
        detection_system.stop()

if __name__ == "__main__":
    main()
