from agent.Base_Agent import Base_Agent
from math_ops.Math_Ops import Math_Ops as M
import math
import numpy as np


class Agent(Base_Agent):
    def __init__(self, host:str, agent_port:int, monitor_port:int, unum:int,
                 team_name:str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:
        
        # define robot type
        robot_type = (0,1,1,1,2,3,3,3,4,4,4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type, team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        self.fat_proxy_walk = np.zeros(3) # filtered walk parameters for fat proxy

        self.init_pos = ([-14,0],[-9,-5],[-9,0],[-9,5],[-5,-5],[-5,0],[-5,5],[-1,-6],[-1,-2.5],[-1,2.5],[-1,6])[unum-1] # initial formation
    
    def predecir_posicion(self, jugador, delta_t=0.5):
        """
        Predice la posición futura de un jugador en delta_t segundos.
        Se asume que el objeto jugador tiene atributos:
          - state_abs_pos: posición actual (lista o array)
          - state_vel: velocidad actual (lista o array)
        """
        # Si no existe la velocidad o la posición es None, retornamos la posición actual o un valor por defecto
        if jugador.state_abs_pos is None:
            return np.array([1000, 1000])
        pos_actual = np.array(jugador.state_abs_pos[:2])
        if not hasattr(jugador, "state_vel") or jugador.state_vel is None:
            return pos_actual
        vel = np.array(jugador.state_vel[:2])
        return pos_actual + vel * delta_t

    def score_pase(self, pred_comp, pred_opp):
        """
        Calcula un score para cada posible pase, evaluando qué tan libre
        está cada compañero de la interferencia de oponentes.
        Se utiliza la distancia promedio de los oponentes a la posición del pase.
        """
        scores = []
        for pos in pred_comp:
            # A mayor distancia de los oponentes, mayor es el score
            dists = [np.linalg.norm(pos - opp) for opp in pred_opp]
            score = np.mean(dists) if dists else 0
            scores.append(score)
        return scores

    def score_tiro(self, ball_pos, goal_pos=(15.05, 0), pred_opp=None):
        """
        Calcula un score para realizar un tiro directo.
        Como ejemplo, se puede usar la distancia del oponente más cercano a la portería,
        de modo que mientras más alejados estén los oponentes, mejor será la opción de tiro.
        """
        if pred_opp is None or not pred_opp:
            return 0
        dists = [np.linalg.norm(np.array(goal_pos) - opp) for opp in pred_opp]
        # Usamos el valor mínimo: a mayor distancia del oponente más cercano, mayor la calidad del tiro
        return np.min(dists)

    def calcular_posicion_segura(self):
        """
        Calcula una posición de reposicionamiento segura.
        Como ejemplo, podríamos tomar el promedio de las posiciones de los oponentes
        y posicionarnos en sentido opuesto al centro de ellos.
        """
        opp_positions = [np.array(p.state_abs_pos[:2]) for p in self.world.opponents if p.state_abs_pos is not None]
        if not opp_positions:
            return self.init_pos  # fallback
        centro = np.mean(opp_positions, axis=0)
        # Posición segura: alejarnos del centro de oponentes, manteniendo cierta coherencia con la formación inicial
        direccion_segura = np.array(self.init_pos) - centro
        return (np.array(self.init_pos) + 0.5 * direccion_segura).tolist()

    def evaluar_opciones(self, ball_pos, teammates, opponents):
        """
        Evalúa y compara tres opciones: pase, tiro o reposicionamiento.
        Retorna una tupla (acción, parámetros) donde acción es una cadena
        que puede ser "pase", "tiro" o "reposicionamiento", y parámetros
        son los datos necesarios para ejecutar la acción.
        """
        # Predecir posiciones futuras para compañeros y oponentes
        pred_comp = [self.predecir_posicion(j) for j in teammates]
        pred_opp = [self.predecir_posicion(j) for j in opponents]

        # Score para pase: queremos que el compañero esté libre de oponentes.
        scores_pase = self.score_pase(pred_comp, pred_opp)
        max_score_pase = max(scores_pase) if scores_pase else -1
        indice_pase = scores_pase.index(max_score_pase) if scores_pase else None

        # Score para tiro: a mayor distancia de oponentes a la portería, mejor.
        score_tiro = self.score_tiro(ball_pos, goal_pos=(15.05, 0), pred_opp=pred_opp)

        # Score para reposicionamiento: podemos evaluar, por ejemplo,
        # la seguridad de nuestra posición en función de la concentración de oponentes.
        # Aquí usamos como score la inversa de la densidad de oponentes cerca de nosotros.
        # Para simplificar, si la media de distancias de oponentes a la posición actual es baja, 
        # el score de reposicionamiento será alto (menos seguro).
        if pred_opp:
            distancias = [np.linalg.norm(np.array(ball_pos) - opp) for opp in pred_opp]
            score_reposicion = np.mean(distancias)
        else:
            score_reposicion = 0

        # Seleccionar la acción con mayor score.
        # Puedes ajustar estos umbrales y combinaciones según pruebas y simulaciones.
        # En este ejemplo, priorizamos el pase si el score es significativamente mayor que el tiro.
        if max_score_pase > score_tiro + 0.5:
            accion = "pase"
            parametros = pred_comp[indice_pase]
        elif score_tiro > max_score_pase:
            accion = "tiro"
            parametros = None  # Aquí se podría calcular la dirección o potencia del tiro
        else:
            accion = "reposicionamiento"
            parametros = self.calcular_posicion_segura()

        return accion, parametros


    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:] # copy position list 
        self.state = 0

        # Avoid center circle by moving the player back 
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3 

        if np.linalg.norm(pos - r.loc_head_position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            self.scom.commit_beam(pos, M.vector_angle((-pos[0],-pos[1]))) # beam to initial position, face coordinate (0,0)
        else:
            if self.fat_proxy_cmd is None: # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else: # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk


    def move(self, target_2d=(0,0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None: # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation, is_orientation_absolute) # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(target_2d - r.loc_head_position[:2])

        self.behavior.execute("Walk", target_2d, True, orientation, is_orientation_absolute, distance_to_final_target) # Args: target, is_target_abs, ori, is_ori_abs, distance





    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball
            
        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.scom.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None: # normal behavior
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort) # Basic_Kick has no kick distance control
        else: # fat proxy behavior
            return self.fat_proxy_kick()


    def think_and_send(self):
            w = self.world
            r = self.world.robot  
            my_head_pos_2d = r.loc_head_position[:2]
            my_ori = r.imu_torso_orientation
            ball_2d = w.ball_abs_pos[:2]
            ball_vec = ball_2d - my_head_pos_2d
            ball_dir = M.vector_angle(ball_vec)
            ball_dist = np.linalg.norm(ball_vec)
            ball_sq_dist = ball_dist * ball_dist # for faster comparisons
            ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])
            behavior = self.behavior
            goal_dir = M.target_abs_angle(ball_2d,(15.05,0))
            path_draw_options = self.path_manager.draw_options
            PM = w.play_mode
            PM_GROUP = w.play_mode_group

            #--------------------------------------- 1. Preprocessing

            # Obtén la posición predicha de la bola
            slow_ball_pos = w.get_predicted_ball_pos(0.5)  # Predicción cuando la velocidad <= 0.5 m/s

            # Definir una posición por defecto para jugadores sin información (se usará para evitar errores)
            default_pos = np.array([1000, 1000])

            # --- Vectorización para compañeros ---
            # Extraer posiciones: si state_abs_pos es None, se usa default_pos
            teammate_positions = np.array([
                np.array(p.state_abs_pos[:2]) if p.state_abs_pos is not None else default_pos
                for p in w.teammates
            ])
            # Extraer otros atributos con comprobación para evitar None
            teammate_last_updates = np.array([
                p.state_last_update if p.state_last_update is not None else 0
                for p in w.teammates
            ])
            teammate_is_self = np.array([p.is_self for p in w.teammates])
            teammate_fallen = np.array([p.state_fallen for p in w.teammates])
            # Incluimos en la máscara también que la posición sea válida
            valid_mask = (
                (teammate_last_updates != 0) &
                (((w.time_local_ms - teammate_last_updates) <= 360) | teammate_is_self) &
                (~teammate_fallen) &
                (np.array([p.state_abs_pos is not None for p in w.teammates]))
            )
            # Calcula las diferencias y la norma al cuadrado
            diff_teammates = teammate_positions - slow_ball_pos
            sq_distances_teammates = np.sum(diff_teammates**2, axis=1)
            # Para los jugadores con datos inválidos, asigna un valor grande (1000)
            sq_distances_teammates[~valid_mask] = 1000
            teammates_ball_sq_dist = sq_distances_teammates.tolist()

            # --- Vectorización para oponentes ---
            opponent_positions = np.array([
                np.array(p.state_abs_pos[:2]) if p.state_abs_pos is not None else default_pos
                for p in w.opponents
            ])
            opponent_last_updates = np.array([
                p.state_last_update if p.state_last_update is not None else 0
                for p in w.opponents
            ])
            opponent_fallen = np.array([p.state_fallen for p in w.opponents])
            valid_mask_opponents = (
                (opponent_last_updates != 0) &
                ((w.time_local_ms - opponent_last_updates) <= 360) &
                (~opponent_fallen) &
                (np.array([p.state_abs_pos is not None for p in w.opponents]))
            )
            diff_opponents = opponent_positions - slow_ball_pos
            sq_distances_opponents = np.sum(diff_opponents**2, axis=1)
            sq_distances_opponents[~valid_mask_opponents] = 1000
            opponents_ball_sq_dist = sq_distances_opponents.tolist()

            # Calcular las distancias mínimas
            min_teammate_ball_sq_dist = np.min(teammates_ball_sq_dist)
            self.min_teammate_ball_dist = math.sqrt(min_teammate_ball_sq_dist)
            self.min_opponent_ball_dist = math.sqrt(np.min(opponents_ball_sq_dist))


            active_player_unum = teammates_ball_sq_dist.index(min_teammate_ball_sq_dist) + 1


            #--------------------------------------- 2. Decide action



            if PM == w.M_GAME_OVER:
                pass
            elif PM_GROUP == w.MG_ACTIVE_BEAM:
                self.beam()
            elif PM_GROUP == w.MG_PASSIVE_BEAM:
                self.beam(True) # avoid center circle
            elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
                self.state = 0 if behavior.execute("Get_Up") else 1 # return to normal state if get up behavior has finished
            elif PM == w.M_OUR_KICKOFF:
                if r.unum == 9:
                    self.kick(120,3) # no need to change the state when PM is not Play On
                else:
                    self.move(self.init_pos, orientation=ball_dir) # walk in place
            elif PM == w.M_THEIR_KICKOFF:
                self.move(self.init_pos, orientation=ball_dir) # walk in place
            elif active_player_unum != r.unum: # I am not the active player
                if r.unum == 1: # I am the goalkeeper
                    self.move(self.init_pos, orientation=ball_dir) # walk in place 
                else:
                    # compute basic formation position based on ball position
                    new_x = max(0.5,(ball_2d[0]+15)/15) * (self.init_pos[0]+15) - 15
                    if self.min_teammate_ball_dist < self.min_opponent_ball_dist:
                        new_x = min(new_x + 3.5, 13) # advance if team has possession
                    self.move((new_x,self.init_pos[1]), orientation=ball_dir, priority_unums=[active_player_unum])


            else: # I am the active player
                path_draw_options(enable_obstacles=True, enable_path=True, use_team_drawing_channel=True) # enable path drawings for active player (ignored if self.enable_draw is False)
                enable_pass_command = (PM == w.M_PLAY_ON and ball_2d[0]<6)
                
                # Evaluación dinámica de opciones para el jugador activo:
                accion, parametros = self.evaluar_opciones(ball_2d, w.teammates, w.opponents)
                if accion == "pase":
                    # Moverse para un pase óptimo (se podría agregar lógica extra para calcular mejor la posición)
                    self.move(parametros, orientation=ball_dir, priority_unums=[active_player_unum])

                elif accion == "tiro":
                    # Ejecutar un tiro directo
                    goal_dir = M.target_abs_angle(ball_2d, (15.05, 0))
                    self.kick(goal_dir, 9, False, enable_pass_command)
                elif accion == "reposicionamiento":
                    # Reposicionarse a una posición más segura
                    self.move(parametros, orientation=ball_dir)
                else:
                    # Moverse hacia la bola si no hay otra acción específica
                    self.move(ball_2d, orientation=ball_dir, avoid_obstacles=False)

                path_draw_options(enable_obstacles=False, enable_path=False) # disable path drawings

            #--------------------------------------- 3. Broadcast
            self.radio.broadcast()

            #--------------------------------------- 4. Send to server
            if self.fat_proxy_cmd is None: # normal behavior
                self.scom.commit_and_send( r.get_command() )
            else: # fat proxy behavior
                self.scom.commit_and_send( self.fat_proxy_cmd.encode() ) 
                self.fat_proxy_cmd = ""

            #---------------------- annotations for debugging
            if self.enable_draw: 
                d = w.draw
                if active_player_unum == r.unum:
                    d.point(slow_ball_pos, 3, d.Color.pink, "status", False) # predicted future 2D ball position when ball speed <= 0.5 m/s
                    d.point(w.ball_2d_pred_pos[-1], 5, d.Color.pink, "status", False) # last ball prediction
                    d.annotation((*my_head_pos_2d, 0.6), "I've got it!" , d.Color.yellow, "status")
                else:
                    d.clear("status")




        #--------------------------------------- Fat proxy auxiliary methods


    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot 
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.loc_head_position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {M.normalize_deg( self.kick_direction  - r.imu_torso_orientation ):.2f} 20)" 
            self.fat_proxy_walk = np.zeros(3) # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1,0), None, True) # ignore obstacles
            return False


    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.loc_head_position[:2])
        target_dir = M.target_rel_angle(r.loc_head_position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = M.normalize_deg( orientation - r.imu_torso_orientation )
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")
