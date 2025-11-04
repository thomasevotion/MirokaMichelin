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

def find_closest_phone_to_person(person_bbox, phones, max_distance=200):
    """Trouve le téléphone le plus proche d'une personne avec distance augmentée"""
    if not phones:
        return None
    
    person_center = ((person_bbox[0] + person_bbox[2]) / 2, 
                    (person_bbox[1] + person_bbox[3]) / 2)
    
    closest_phone = None
    min_distance = float('inf')
    
    for phone in phones:
        phone_center = ((phone['bbox'][0] + phone['bbox'][2]) / 2,
                       (phone['bbox'][1] + phone['bbox'][3]) / 2)
        
        distance = np.sqrt((person_center[0] - phone_center[0])**2 + 
                         (person_center[1] - phone_center[1])**2)
        
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]
        
        # Zone étendue pour associer les téléphones
        extended_x1 = person_bbox[0] - person_width * 0.8
        extended_y1 = person_bbox[1] - person_height * 0.8
        extended_x2 = person_bbox[2] + person_width * 0.8
        extended_y2 = person_bbox[3] + person_height * 0.8
        
        phone_x1, phone_y1, phone_x2, phone_y2 = phone['bbox']
        phone_in_zone = (extended_x1 <= phone_x1 <= extended_x2 and 
                        extended_x1 <= phone_x2 <= extended_x2 and
                        extended_y1 <= phone_y1 <= extended_y2 and 
                        extended_y1 <= phone_y2 <= extended_y2)
        
        if distance < min_distance and distance < max_distance and phone_in_zone:
            min_distance = distance
            closest_phone = phone
    
    return closest_phone

def is_phone_near_head(person_bbox, phone_bbox):
    """Détection du téléphone près de la tête avec seuils plus permissifs"""
    person_x1, person_y1, person_x2, person_y2 = person_bbox
    phone_x1, phone_y1, phone_x2, phone_y2 = phone_bbox
    
    # Zone de tête étendue (60% de la hauteur au lieu de 50%)
    head_y1 = person_y1
    head_y2 = person_y1 + (person_y2 - person_y1) * 0.6
    
    phone_center_y = (phone_y1 + phone_y2) / 2
    phone_center_x = (phone_x1 + phone_x2) / 2
    person_center_x = (person_x1 + person_x2) / 2
    
    person_width = person_x2 - person_x1
    # Distance horizontale plus permissive (80% au lieu de 60%)
    max_horizontal_distance = person_width * 0.8
    
    in_head_zone = head_y1 <= phone_center_y <= head_y2
    close_horizontally = abs(phone_center_x - person_center_x) < max_horizontal_distance
    
    # Seuil "très proche" plus permissif (50% au lieu de 30%)
    phone_very_close = (phone_center_y >= head_y1 and phone_center_y <= head_y2 and
                       abs(phone_center_x - person_center_x) < person_width * 0.5)
    
    return (in_head_zone and close_horizontally) or phone_very_close

def detect_movement_stable(displacement_history, frame_count):
    """Détection de mouvement stable avec seuils adaptatifs"""
    if len(displacement_history) < 3:
        return False
    
    recent_displacements = displacement_history[-5:]
    
    # Seuils adaptatifs basés sur la variance
    variance = np.var(recent_displacements)
    if variance < 0.5:  # Très stable
        threshold = 4.0
    elif variance < 2.0:  # Modérément stable
        threshold = 3.0
    else:  # Instable
        threshold = 2.0
    
    avg_recent = np.mean(recent_displacements)
    if avg_recent < threshold:
        return False
    
    significant_movements = [d for d in recent_displacements if d > threshold * 0.7]
    if len(significant_movements) < 3:
        return False
    
    if len(recent_displacements) >= 4:
        last_three = recent_displacements[-3:]
        if all(d > threshold * 0.5 for d in last_three):
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
        
        # Lissage temporel pour téléphone
        self.person_phone_history = {}  # Historique des détections téléphone (dernières N frames)
        self.person_phone_state = {}    # État stable de téléphone (lissé)
        self.phone_history_size = 10    # Nombre de frames à garder dans l'historique
        self.phone_activation_threshold = 0.4  # 40% des frames doivent avoir téléphone pour activer
        self.phone_deactivation_threshold = 0.3  # Moins de 30% pour désactiver (hystérésis)
        
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
    
    def update_persons(self, detected_persons, phones, frame_count):
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
                if len(self.person_movement_history[person_id]) > 15:
                    self.person_movement_history[person_id] = self.person_movement_history[person_id][-15:]
                
                # Détection de mouvement stable
                is_moving_detected = detect_movement_stable(self.person_movement_history[person_id], frame_count)
                
                # État de mouvement persistant
                if is_moving_detected:
                    self.person_movement_state[person_id] = True
                else:
                    if self.person_movement_state[person_id]:
                        recent_displacements = self.person_movement_history[person_id][-5:]
                        if all(d < 1.5 for d in recent_displacements):
                            self.person_movement_state[person_id] = False
                
                is_moving = self.person_movement_state[person_id]
                
                # Trouver le téléphone le plus proche (détection instantanée)
                closest_phone = find_closest_phone_to_person(person_bbox, phones)
                current_phone_detected = False
                phone_bbox_for_state = None
                
                if closest_phone:
                    # Vérifier si le téléphone est vraiment associé à cette personne
                    if is_phone_near_head(person_bbox, closest_phone['bbox']) or \
                       calculate_iou(person_bbox, closest_phone['bbox']) > 0.1:
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
                
                # Trouver le téléphone le plus proche (détection instantanée)
                closest_phone = find_closest_phone_to_person(person_bbox, phones)
                current_phone_detected = False
                phone_bbox_for_state = None
                
                if closest_phone:
                    # Vérifier si le téléphone est vraiment associé à cette personne
                    if is_phone_near_head(person_bbox, closest_phone['bbox']) or \
                       calculate_iou(person_bbox, closest_phone['bbox']) > 0.1:
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
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.latest_stats = {}
        
    def initialize(self):
        """Initialise le système de détection avec modèle YOLOv8m"""
        try:
            # Charger le modèle YOLOv8m directement
            logger.info("🔄 Chargement du modèle YOLOv8m...")
            
            try:
                self.model = YOLO('yolov8m.pt')
                logger.info("✅ Modèle YOLOv8m chargé! (Très précis pour téléphones)")
            except:
                # Fallback vers YOLOv8s si m n'est pas disponible
                try:
                    self.model = YOLO('yolov8s.pt')
                    logger.info("✅ Modèle YOLOv8s chargé! (Fallback)")
                except:
                    # Fallback vers nano si rien d'autre n'est disponible
                    self.model = YOLO('yolov8n.pt')
                    logger.warning("⚠️ Utilisation du modèle nano (fallback)")
            
            # Ouvrir la caméra
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Impossible d'ouvrir la caméra")
            
            # Configuration optimisée
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 60)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            logger.info("✅ Caméra ouverte!")
            logger.info("📹 Configuration: 640x480 @ 60fps")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation: {e}")
            return False
    
    def process_frame(self, frame):
        """Traite une frame avec seuils optimisés pour téléphones"""
        if self.model is None:
            return [], []
        
        # Détection YOLO avec seuils optimisés et NMS plus stricte
        # - iou plus élevé pour réduire les doublons
        # - max_det limité pour éviter la sur-détection
        # - conf global bas, filtrage par classe ensuite
        results = self.model(frame, conf=0.15, iou=0.6, max_det=50, classes=[0, 67], verbose=False)
        
        persons = []
        phones = []
        
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                if class_id == 0:  # person
                    # Seuil plus strict pour réduire les faux positifs
                    if confidence >= 0.50:
                        persons.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence
                        })
                elif class_id == 67:  # cell phone
                    # Seuil très bas pour les téléphones
                    if confidence >= 0.15:
                        phones.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence
                        })
        
        return persons, phones
    
    def detection_loop(self):
        """Boucle principale de détection avec visualisation"""
        frame_count = 0
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("⚠️ Impossible de lire la frame")
                    continue
                
                frame_count += 1
                
                # Traiter la frame
                persons, phones = self.process_frame(frame)
                
                # Mettre à jour le suivi des personnes
                current_persons = self.tracker.update_persons(persons, phones, frame_count)
                
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
                
                # Log périodique
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
    print("🚀 SYSTÈME DE DÉTECTION AMÉLIORÉ - TÉLÉPHONES")
    print("=" * 60)
    print("📱 Améliorations pour la détection de téléphones:")
    print("  - Modèle YOLOv8m (Medium - Très précis)")
    print("  - Seuil de confiance abaissé (0.15 pour téléphones)")
    print("  - Zone d'association étendue")
    print("  - Détection de tête plus permissive")
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
        print("\n⏹️ Arrêt demandé")
    finally:
        detection_system.stop()

if __name__ == "__main__":
    main()
