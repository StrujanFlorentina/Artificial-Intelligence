(define (problem chernobyl-operation-p)
  (:domain chernobyl-operation)
  (:objects
    z0 z1 z2 z3 z4 z5 - zone
    v0 v1 - volunteer
    a0 a1 - ambulance
    mat0 mat1 mat2 mat3 mat4 mat5 mat6 - radioactiveMaterial
    zero one two three  - quantity
  )

  (:init
    (available v0)
    (available v1)
    (prezent a0)
    (capacity v0 three)
    (capacity v1 three)
    (localized mat0)
    (next-material mat0 mat1)
    (next-material mat1 mat2)
    (next-material mat2 mat3)
    (next-material mat3 mat4)
    (next-material mat4 mat5)
    (next-material mat5 mat6)
    (reactor-entrance z0)
    (connected z0 z1)
    (connected z1 z0)
    (connected z1 z2)
    (connected z2 z1)
    (connected z2 z3)
    (connected z3 z2)
    (connected z0 z4)
    (connected z4 z0)
    (connected z4 z5)
    (connected z5 z4)
    (next-quantity zero one)
    (next-quantity one two)
    (next-quantity two three)
    (next-quantity three one)
        (= (total-cost) 0)
  )

  (:goal
    (and
      (clean z5)
      (extra-health-care v1)
      (health-care v0)
    )
  )
    (:metric minimize (total-cost))


)
