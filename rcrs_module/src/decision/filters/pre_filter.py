from __future__ import annotations

import logging
from typing import Iterable, List, Optional, TYPE_CHECKING

from decision.utility.distance import MAX_MAP_DISTANCE
from world.entities import AgentState, AgentType, EntityType, VisibleEntity

if TYPE_CHECKING:
    from world.cache import WorldModel


logger = logging.getLogger(__name__)

ESTIMATED_TRIP_TO_REFUGE: float = 10.0

UNREACHABLE_DISTANCE_THRESHOLD: float = MAX_MAP_DISTANCE * 0.9


class NeedRefugeException(RuntimeError):
    """пожарному нужно направиться в убежище."""


class PreFilterDispatcher:
    def __init__(self, work_rate: float = 1.0, average_speed: float = 70_000.0) -> None:
        if work_rate <= 0:
            raise ValueError("Я ожидаю положительный work_rate для расчета дедлайнов")
        if average_speed <= 0:
            raise ValueError("Я ожидаю положительный average_speed для конвертации дистанции")
        self.work_rate = work_rate
        self.average_speed = average_speed

    def _allowed_entity_types(self, agent_type: AgentType) -> frozenset[EntityType]:
        if agent_type == AgentType.FIRE_BRIGADE:
            # Я разрешаю пожарному не только тушить здания, но и откапывать
            # заваленных людей (AKRescue): сервер это поддерживает
            # (MiscSimulator.checkRescue принимает FireBrigade).
            return frozenset({EntityType.BUILDING, EntityType.CIVILIAN, EntityType.HUMAN})
        if agent_type == AgentType.AMBULANCE_TEAM:
            return frozenset({EntityType.CIVILIAN, EntityType.HUMAN})
        return frozenset({EntityType.BLOCKADE})

    def filter_tasks(
        self,
        agent_state: AgentState,
        tasks: Iterable[VisibleEntity],
        world_model: Optional["WorldModel"] = None,
    ) -> List[VisibleEntity]:

        try:
            if agent_state.resources.is_transporting:
                logger.debug(
                    "Я пропускаю фильтрацию, потому что агент занят: agent_id=%s",
                    agent_state.id,
                )
                return []

            allowed_types = self._allowed_entity_types(agent_state.type)
            tasks_list = list(tasks)
            tasks = [e for e in tasks_list if e.type in allowed_types]

            # Без воды пожарный не может тушить — убираю здания из пула задач,
            # оставляя только спасение заваленных. Раньше здесь сразу
            # выбрасывалось NeedRefugeException, из-за чего пожарный уезжал в
            # убежище, игнорируя заваленных мирных рядом.
            if agent_state.type == AgentType.FIRE_BRIGADE and agent_state.resources.water_quantity == 0:
                tasks = [e for e in tasks if e.type != EntityType.BUILDING]
                if not tasks:
                    logger.debug(
                        "Я направляю пожарного в убежище: воды нет и откапывать некого, agent_id=%s",
                        agent_state.id,
                    )
                    raise NeedRefugeException("Я обнаружил, что у пожарного нет воды и задач на спасение")
            logger.debug(
                "Я отфильтровал сущности по типам %s: было=%d, осталось=%d, agent_id=%s",
                ",".join(t.value for t in sorted(allowed_types, key=lambda t: t.value)),
                len(tasks_list),
                len(tasks),
                agent_state.id,
            )

            has_live_fire_brigades = self._has_live_fire_brigades(world_model)
            refuge_set = self._refuge_set(world_model)

            filtered: List[VisibleEntity] = []
            for entity in tasks:
                try:
                    if self._is_relevant(
                        entity,
                        agent_type=agent_state.type,
                        has_live_fire_brigades=has_live_fire_brigades,
                        refuge_set=refuge_set,
                    ):
                        filtered.append(entity)
                except ValueError as exc:
                    logger.error(
                        "Я получил некорректные данные для entity_id=%s: %s",
                        entity.id,
                        exc,
                    )

            if not filtered and tasks:
                logger.info(
                    "ДИАГ_ФИЛЬТР: %d задач → 0 после фильтра, agent_id=%s, типы: %s",
                    len(tasks),
                    agent_state.id,
                    ", ".join(
                        f"{e.id}(f={e.raw_sensor_data.fieryness},hp={e.raw_sensor_data.hp},"
                        f"d={e.raw_sensor_data.damage},b={e.raw_sensor_data.buriedness},"
                        f"pd={e.computed_metrics.path_distance:.0f})"
                        for e in list(tasks)[:5]
                    ),
                )

            return filtered
        except ValueError as exc:
            logger.error("Я получил некорректные входные данные фильтра: %s", exc)
            return []

    @staticmethod
    def _has_live_fire_brigades(world_model: Optional["WorldModel"]) -> bool:
        if world_model is None:
            return True
        for ally in world_model.agents.values():
            if ally.type != AgentType.FIRE_BRIGADE:
                continue
            if ally.hp is None or ally.hp > 0:
                return True
        return False

    @staticmethod
    def _refuge_set(world_model: Optional["WorldModel"]) -> frozenset[int]:
        if world_model is None:
            return frozenset()
        return frozenset(world_model.refuge_ids)

    def _is_relevant(
        self,
        entity: VisibleEntity,
        agent_type: AgentType,
        has_live_fire_brigades: bool,
        refuge_set: frozenset[int],
    ) -> bool:
        hp = entity.raw_sensor_data.hp
        damage = entity.raw_sensor_data.damage
        buriedness = entity.raw_sensor_data.buriedness
        pos_edge = entity.raw_sensor_data.position_on_edge

        if entity.type == EntityType.CIVILIAN:
            if hp is not None and hp == 0:
                logger.debug("Я исключаю погибшего гражданского: entity_id=%s", entity.id)
                return False

            if damage == 0 and buriedness == 0:
                logger.debug(
                    "Я исключаю здорового гражданского без завала: entity_id=%s",
                    entity.id,
                )
                return False

            # Гражданских с неизвестным состоянием (damage=None, buriedness=None)
            # НЕ отсеиваем: данные обновятся из ChangeSet при подходе агента.
            # Urgency вернёт 0 для неизвестных, но агент будет двигаться к ним,
            # а не блуждать случайно.

            # Гражданский, уже оказавшийся в убежище, — не наша цель.
            if pos_edge is not None and pos_edge in refuge_set:
                logger.debug(
                    "Я исключаю гражданского в убежище: entity_id=%s, pos=%s",
                    entity.id, pos_edge,
                )
                return False

            # Пожарный спасает ТОЛЬКО заваленных и раненых: damage>0 и
            # buriedness>0 (аналог `_is_valid_human` в ADF). Здоровых
            # гражданских пропускаем — это зона ответственности амбуланса.
            if agent_type == AgentType.FIRE_BRIGADE:
                if damage is None or damage <= 0:
                    return False
                if buriedness is None or buriedness <= 0:
                    return False

            # Амбуланс берёт ТОЛЬКО откопанных (buriedness == 0) —
            # заваленных откапывают пожарные. Исключение: если на карте
            # нет живых пожарных вообще, амбуланс берёт и заваленных
            # (лучше криво, чем никак).
            if agent_type == AgentType.AMBULANCE_TEAM:
                if (
                    buriedness is not None
                    and buriedness > 0
                    and has_live_fire_brigades
                ):
                    logger.debug(
                        "Я исключаю заваленного из задач амбуланса "
                        "(это для пожарных): entity_id=%s, buriedness=%s",
                        entity.id, buriedness,
                    )
                    return False

        if entity.type == EntityType.HUMAN:
            if hp is not None and hp == 0:
                logger.debug("Я исключаю погибшего агента: entity_id=%s", entity.id)
                return False

            if buriedness is None or buriedness == 0:
                logger.debug(
                    "Я исключаю незаваленного агента: entity_id=%s, buriedness=%s",
                    entity.id,
                    buriedness,
                )
                return False

            # Союзник в убежище — не спасаем (уже безопасен).
            if pos_edge is not None and pos_edge in refuge_set:
                return False

            # Пожарный: заваленному союзнику нужны dmg>0 И buriedness>0.
            if agent_type == AgentType.FIRE_BRIGADE:
                if damage is None or damage <= 0:
                    return False

            # Амбуланс: только откопанные (иначе — дело пожарных).
            if agent_type == AgentType.AMBULANCE_TEAM and has_live_fire_brigades:
                return False

        if entity.type == EntityType.BUILDING:
            fieryness = entity.raw_sensor_data.fieryness
            # Я пропускаю только активно горящие здания (fieryness 1-3) и
            # здания с неизвестным состоянием (fieryness=None, стоит проверить).
            # Все остальные (0 — целое, 4-7 — потушено/охлаждается,
            # 8 — полностью сгорело) отсеиваю: тушить нечего, а нулевая
            # urgency при f_dist=0 создавала бесконечный цикл AKRest, из-за
            # которого пожарные стояли на месте вместо исследования карты.
            if fieryness is not None and fieryness not in {1, 2, 3}:
                logger.debug(
                    "Я исключаю неактивное здание: fieryness=%s, entity_id=%s",
                    fieryness, entity.id,
                )
                return False

        if entity.type == EntityType.BLOCKADE:
            repair_cost = entity.raw_sensor_data.repair_cost
            # Я отсеиваю завал ТОЛЬКО если repair_cost явно 0 — это значит
            # «уже расчищено». None (сервер ещё не прислал значение для
            # удалённого наблюдаемого завала) оставляю: такой завал может
            # блокировать важную цель, а с f_effort=0 он всё равно не станет
            # топ-приоритетом сам по себе.
            if repair_cost == 0:
                logger.debug(
                    "Я исключаю завал: repair_cost=0 (расчищено): entity_id=%s",
                    entity.id,
                )
                return False

        if entity.computed_metrics.path_distance >= UNREACHABLE_DISTANCE_THRESHOLD:
            logger.info(
                "Я исключаю недостижимую цель entity_id=%s: path_distance=%.0f ≥ %.0f (нет прохода)",
                entity.id,
                entity.computed_metrics.path_distance,
                UNREACHABLE_DISTANCE_THRESHOLD,
            )
            return False

        if entity.raw_sensor_data.buriedness is not None:
            if self.work_rate <= 0:
                raise ValueError("Я получил неположительную скорость работ")

            try:
                t_travel = entity.computed_metrics.path_distance / self.average_speed
            except ZeroDivisionError:
                t_travel = 0.0

            time_to_action = t_travel + (
                entity.raw_sensor_data.buriedness / self.work_rate
            )
            if entity.computed_metrics.estimated_death_time < (
                time_to_action + ESTIMATED_TRIP_TO_REFUGE
            ):
                logger.debug(
                    "Я исключаю обреченного (не довезет живым): entity_id=%s, "
                    "estimated_death=%d, time_to_action=%.1f, trip_to_refuge=%.1f",
                    entity.id,
                    entity.computed_metrics.estimated_death_time,
                    time_to_action,
                    ESTIMATED_TRIP_TO_REFUGE,
                )
                return False

        return True


__all__ = ["NeedRefugeException", "PreFilterDispatcher"]
