(define (domain chernobyl-operation)
    (:requirements :strips :action-costs)
    (:types zone volunteer ambulance radioactiveMaterial quantity)
    (:predicates
        (localized ?mat - radioactiveMaterial)
        (full ?mat - radioactiveMaterial)
        (capacity ?v - volunteer ?q - quantity)
        (next-material ?mat1 - radioactiveMaterial ?mat2 - radioactiveMaterial)
        (at-volunteer ?v - volunteer ?z - zone)
        (at-material ?mat - radioactiveMaterial ?z - zone)
        (available ?v - volunteer)
        (in-reactor)
        (reactor-entrance ?z - zone)
        (out ?v - volunteer)
        (health-care ?v - volunteer)	
        (extra-health-care ?v - volunteer)	
        (connected ?z1 - zone ?z2 - zone)
        (precludes ?v1 - volunteer ?v2 - volunteer)
        (next-quantity ?q1 - quantity ?q2 - quantity)
        (holding ?v - volunteer ?mat - radioactiveMaterial)
        (clean ?z - zone)
        (occupied ?a - ambulance ?v - volunteer)
        (prezent ?a-ambulance)
        (exhausted ?v - volunteer)
    )
    
    (:functions
        (total-cost) - number
    )
    
    (:action call-volunteer
        :parameters (?v1 - volunteer)
        :precondition   (and(available ?v1)
                            (not (in-reactor)) 
                        )
        :effect (and(out ?v1)
                    (not (available ?v1))
                    (forall (?v2 - volunteer)
                    (when (precludes ?v1 ?v2) (not (available ?v2))))
                    (in-reactor)
                    (increase (total-cost) 1)
                )
    )
    
    (:action call-ambulance
        :parameters (?a1 ?a2 - ambulance ?v - volunteer)
        :precondition   (and(occupied ?a2 ?v )
                            (not (prezent ?a1)) 
                        )
        :effect (and(prezent ?a1)
                    (increase (total-cost) 1)
                )
    )
    
    (:action localize-radioactive-material
        :parameters (?v - volunteer ?mat1 ?mat2 - radioactiveMaterial ?q1 ?q2 - quantity)
        :precondition   (and(out ?v)
                            (localized ?mat1)
                            (next-quantity ?q1 ?q2)
                            (capacity ?v ?q2)
                            (next-material ?mat1 ?mat2)
                        )
        :effect (and(not (localized ?mat1))
                    (not (capacity ?v ?q2))
                    (localized ?mat2)
                    (full ?mat1)
                    (capacity ?v ?q1)
                    (holding ?v ?mat1)
                    (increase (total-cost) 1)

                )
    )

    (:action enter-reactor
        :parameters (?v - volunteer ?z - zone)
        :precondition   (and(out ?v)
                            (reactor-entrance ?z)
                        )
        :effect (and(not (out ?v))
                    (at-volunteer ?v ?z)
                    (increase (total-cost) 1)
                )
    )

    (:action take-radioactive-material
        :parameters (?v - volunteer ?mat - radioactiveMaterial ?z - zone ?q1 ?q2 - quantity)
        :precondition   (and(at-volunteer ?v ?z)
                            (at-material ?mat ?z)
                            (next-quantity ?q1 ?q2)
                            (capacity ?v ?q2)
                        )
        :effect (and(not (at-material ?mat ?z))
                    (not (capacity ?v ?q2))
                    (holding ?v ?mat)
                    (capacity ?v ?q1)
                    (increase (total-cost) 1)
                )
    )

    (:action take-localized-radioactive-material
        :parameters (?v - volunteer ?mat - radioactiveMaterial ?z - zone ?q1 ?q2 - quantity)
        :precondition   (and(at-volunteer ?v ?z)
                            (holding ?v ?mat)
                            (next-quantity ?q1 ?q2)
                            (capacity ?v ?q1)
                        )
        :effect (and(not (holding ?v ?mat))
                    (not (capacity ?v ?q1))
                    (at-material ?mat ?z)
                    (capacity ?v ?q2)
                    (increase (total-cost) 1)
                )
    )

    (:action walk
        :parameters (?v - volunteer ?mat - radioactiveMaterial ?z1 ?z2 - zone)
        :precondition   (and(at-volunteer ?v ?z1)
                            (holding ?v ?mat)
                            (full ?mat)
                            (connected ?z1 ?z2)
                        )
        :effect (and(not (at-volunteer ?v ?z1))
                    (not (full ?mat))
                    (at-volunteer ?v ?z2)
                    (increase (total-cost) 1)
                )
    )

    (:action clean-zone
        :parameters (?v - volunteer ?z - zone ?mat - radioactiveMaterial)
        :precondition   (and(at-volunteer ?v ?z)
                            (holding ?v ?mat)
                            (full ?mat)
                        )
        :effect (and(not (full ?mat))
                    (clean ?z)
                    (increase (total-cost) 1)
                )
    )

    (:action clean-dangerous-zone
        :parameters (?v - volunteer ?z - zone ?mat - radioactiveMaterial)
        :precondition   (and(at-volunteer ?v ?z)
                            (holding ?v ?mat)
                            (full ?mat)
                        )
        :effect (and(not (full ?mat))
                    (clean ?z)
                    (exhausted ?v)
                    (increase (total-cost) 1)
                )
    )
    
    (:action get-medical-attention
        :parameters (?v - volunteer ?z - zone)
        :precondition   (and(at-volunteer ?v ?z)
                            (reactor-entrance ?z)
                            (not(exhausted ?v))
                        )
        :effect (and(not (at-volunteer ?v ?z))
                    (health-care ?v)
                    (not (in-reactor))
                    (increase (total-cost) 1)
                )
    )
    
    (:action take-to-hospital
        :parameters (?v1 ?v2 - volunteer ?a - ambulance ?z - zone)
        :precondition   (and(at-volunteer ?v1 ?z)
                            (exhausted ?v1)
                            (prezent ?a)
                            (not(occupied ?a ?v2)
                            )
                        )
        :effect (and(not (at-volunteer ?v1 ?z))
                    (extra-health-care ?v1)
                    (not (in-reactor))
                    (not (prezent ?a))
                    (occupied ?a ?v1)
                    (increase (total-cost) 1)
                )
    )
    
)
