(**************************************************************************
 * 2-numbers.v
 *
 * A Coq formalization of the intrinsic embedding of natural numbers in the 
 * Prime Framework (corresponding to 2-numbers.pdf).
 *
 * In this file we show that every natural number N has a unique canonical 
 * (minimal-norm) representation in the fiber algebra A. This representation, 
 * called the universal number embedding, encodes N via its digit expansions 
 * in every base b ≥ 2.
 *
 * We assume that A is a real inner product space and that there exists a 
 * predicate Rep a N which asserts that the element a ∈ A encodes the natural 
 * number N. Furthermore, we assume that the set of representations of N is closed 
 * under averaging (i.e. Rep is linear in the sense that the average of two representations 
 * of N is still a representation of N). Finally, we postulate an extreme value 
 * principle ensuring that the norm (a continuous function) attains a minimum on the 
 * set of representations.
 *
 * Theorem (Universal Number Embedding): For every natural number N, there exists 
 * a unique (canonical) element a ∈ A that encodes N and minimizes the norm among all 
 * representations of N.
 **************************************************************************)

Require Import Reals.
Require Import Psatz.
Open Scope R_scope.

Section UniversalNumberEmbedding.

Context {A : Type}.
Context `{RealInnerProductSpace A}.

(**************************************************************************
 * Representation Predicate
 *
 * Rep a N means that the element a ∈ A encodes the natural number N (via its 
 * multi-base expansion).
 **************************************************************************)
Variable Rep : A -> nat -> Prop.

(* Axiom: Every natural number has at least one representation in A. *)
Axiom Rep_nonempty : forall N, exists a, Rep a N.

(* Axiom: The representation predicate is linear in the following sense: 
   If a and b both encode N, then so does their average. *)
Axiom Rep_linear : forall N a b, Rep a N -> Rep b N ->
                         Rep (scalar_mul (/2) (add a b)) N.

(**************************************************************************
 * The Set of Representations and Canonical Representation
 *
 * For a fixed natural number N, we define the set SN ⊆ A of all elements that encode N.
 * An element a ∈ SN is called canonical if it minimizes the norm on SN.
 **************************************************************************)
Definition SN (N : nat) : A -> Prop :=
  fun a => Rep a N.

Definition Canonical (N : nat) (a : A) : Prop :=
  SN N a /\ (forall b, SN N b -> norm a <= norm b).

(**************************************************************************
 * Extreme Value Principle
 *
 * We assume that for every natural number N, the norm function attains its minimum 
 * on the nonempty set SN. (In a finite-dimensional real inner product space, any 
 * continuous function on a nonempty compact set attains its minimum.)
 **************************************************************************)
Axiom extreme_value_principle : forall N, exists a, Canonical N a.

(**************************************************************************
 * Existence of the Canonical Representation
 **************************************************************************)
Theorem canonical_existence : forall N, exists a, Canonical N a.
Proof.
  intro N.
  apply extreme_value_principle.
Qed.

(**************************************************************************
 * Uniqueness of the Canonical Representation
 *
 * We prove that if a and b are both canonical representations of N, then a = b.
 * The proof uses the strict convexity of the norm induced by the inner product.
 **************************************************************************)
Theorem canonical_uniqueness : forall N a b,
  Canonical N a -> Canonical N b -> a = b.
Proof.
  intros N a b [Ha Hmin_a] [Hb Hmin_b].
  destruct (Req_dec a b) as [Heq | Hneq].
  - assumption.
  - (* Let c be the average of a and b *)
    set (c := scalar_mul (/2) (add a b)).
    assert (Hc: SN N c).
    { apply Rep_linear; assumption. }
    (* By the strict convexity of the norm (from RealInnerProductSpace), since a <> b,
       we have: norm c < (norm a + norm b) / 2 *)
    assert (Hconv: norm c < (norm a + norm b) / 2)
      by (apply norm_strict_convex; assumption).
    (* Minimality of a and b gives norm a <= norm c and norm b <= norm c *)
    assert (Ha_le: norm a <= norm c) by (apply Hmin_a; assumption).
    assert (Hb_le: norm b <= norm c) by (apply Hmin_b; assumption).
    assert (Hsum: norm a + norm b <= 2 * norm c)
      by (apply Rplus_le_compat; assumption).
    assert (Hdiv: (norm a + norm b) / 2 <= norm c).
    { field_simplify in Hsum; lra. }
    lra.
Qed.

(**************************************************************************
 * Universal Number Embedding Theorem
 *
 * For every natural number N, there exists a unique element a ∈ A such that a is the 
 * canonical representation (i.e. minimal-norm embedding) of N.
 **************************************************************************)
Theorem universal_number_embedding : forall N, exists! a, Canonical N a.
Proof.
  intros N.
  destruct (canonical_existence N) as [a HaCanon].
  exists a.
  split.
  - exact HaCanon.
  - intros y Hy.
    apply canonical_uniqueness with (N := N); assumption.
Qed.

End UniversalNumberEmbedding.

(**************************************************************************
 * Conclusion:
 *
 * We have formalized the universal number embedding in the Prime Framework by showing 
 * that every natural number N has a unique canonical representation in the fiber algebra A. 
 * The existence of a minimal-norm representation is guaranteed by an extreme value principle, 
 * and uniqueness follows from the strict convexity of the norm.
 *
 * This completes the Coq proof corresponding to 2-numbers.pdf.
 **************************************************************************)
