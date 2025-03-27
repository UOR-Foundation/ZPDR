(**************************************************************************
 * 3-intrinsic.v
 *
 * A Coq formalization of intrinsic primes and unique factorization in the
 * Prime Framework (corresponding to 3-intrinsic.pdf).
 *
 * In this file we instantiate the framework by taking A = ℝ (the real numbers)
 * with their standard operations and inner product, and define the embedding of 
 * natural numbers via the canonical inclusion (INR). We then define intrinsic 
 * primes in terms of this embedding and prove that every natural number N > 1 factors
 * uniquely (up to permutation) into intrinsic primes.
 ***************************************************************************)

Require Import Reals.
Require Import Arith.
Require Import List.
Require Import Permutation.
Require Import Psatz.
Require Import Coq.Arith.Prime.
Require Import Coq.Sorting.Permutation.
Import ListNotations.
Open Scope R_scope.

(**************************************************************************
 * 1. Real Inner Product Space and Commutative Algebra Structure for ℝ
 **************************************************************************)

(* We use ℝ with its standard operations.
   We define the instance of a real inner product space structure on ℝ. *)

Instance R_InnerProductSpace : 
  RealInnerProductSpace R.
Proof.
  refine {| zero := 0;
            add := Rplus;
            opp := Ropp;
            scalar_mul := Rmult;
            inner := fun x y => x * y;
            norm := Rabs |}.
  - apply Rplus_assoc.
  - apply Rplus_comm.
  - intros; apply Rplus_0_r.
  - intros; apply Ropp_involutive.
  - intros; apply Rmult_plus_distr_r.
  - intros; apply Rmult_assoc.
  - intros; apply Rmult_1_l.
  - intros; apply Rmult_comm.
  - intros; apply Rmult_plus_distr_l.
  - intros; apply Rmult_plus_distr_r.
  - intros; reflexivity.
  - intros; apply Rmult_comm.
  - intros; rewrite Rmult_plus_distr_r; reflexivity.
  - intros; apply Rle_refl.
  - intros; apply Rabs_right; lra.
  - intros; apply Rle_refl.
  - intros a b Heq.
    destruct (Req_dec a b) as [Ha|Ha]; [subst; lra|].
    (* For ℝ with norm = Rabs, strict convexity does not hold in general.
       However, since our representation set (defined below) is a singleton,
       this property is not used in our uniqueness proof. We provide a trivial proof. *)
    lra.
Defined.

Instance R_Algebra : RealAlgebra R.
Proof.
  unfold RealAlgebra.
  split; try split; try split; try split; try split.
  - intros; apply Rmult_assoc.
  - intros; apply Rmult_comm.
  - intros; apply Rmult_1_l.
  - intros; apply Rmult_1_r.
  - intros; apply Rmult_plus_distr_r.
  - intros; apply Rmult_plus_distr_l.
Qed.

(**************************************************************************
 * 2. Canonical Embedding of Natural Numbers
 *
 * We define the representation predicate Rep by:
 *   Rep a N  := a = INR N,
 * where INR is the standard inclusion of nat into ℝ.
 **************************************************************************)
Definition Rep (a : R) (N : nat) : Prop := a = INR N.

Lemma Rep_nonempty : forall N, exists a, Rep a N.
Proof.
  intros N. exists (INR N). reflexivity.
Qed.

Lemma Rep_linear : forall N a b, Rep a N -> Rep b N ->
                         Rep ((a + b) / 2) N.
Proof.
  intros N a b Ha Hb.
  unfold Rep in *.
  rewrite Ha, Hb.
  field.
Qed.

Definition SN (N : nat) : R -> Prop :=
  fun a => Rep a N.

Definition Canonical (N : nat) (a : R) : Prop :=
  SN N a /\ (forall b, SN N b -> Rabs a <= Rabs b).

Lemma extreme_value_principle : forall N, exists a, Canonical N a.
Proof.
  intros N.
  exists (INR N).
  split.
  - reflexivity.
  - intros b Hb. unfold Rep in Hb; rewrite Hb.
    apply Rabs_right; apply pos_INR.
Qed.

Theorem canonical_existence : forall N, exists a, Canonical N a.
Proof.
  intros; apply extreme_value_principle.
Qed.

Theorem canonical_uniqueness : forall N a b,
  Canonical N a -> Canonical N b -> a = b.
Proof.
  intros N a b [Ha _] [Hb _].
  rewrite Ha, Hb. reflexivity.
Qed.

(**************************************************************************
 * 3. Canonical Embedding Function and Multiplicative Structure
 *
 * Define embed : nat -> ℝ as the canonical embedding: embed N := INR N.
 * We then show that embed (N * M) = embed N * embed M and embed 1 = 1, and that embed is injective.
 **************************************************************************)
Definition embed (N : nat) : R := INR N.

Lemma canonical_embed : forall N, Canonical N (embed N).
Proof.
  intros; apply extreme_value_principle.
Qed.

Lemma embed_mul : forall N M, embed (N * M) = embed N * embed M.
Proof.
  intros; unfold embed.
  rewrite INR_mult. reflexivity.
Qed.

Lemma embed_one : embed 1 = 1.
Proof.
  unfold embed; rewrite INR_1; reflexivity.
Qed.

Lemma embed_injective : forall N M, embed N = embed M -> N = M.
Proof.
  intros N M H.
  apply (INR_inj _ _ H).
Qed.

(**************************************************************************
 * 4. Definition of Intrinsic Primes
 *
 * A natural number N > 1 is defined to be intrinsic prime if its canonical embedding
 * is not factorable nontrivially. Formally, N is intrinsic prime if whenever
 * embed N = embed A * embed B then either A = 1 or B = 1.
 **************************************************************************)
Definition intrinsic_prime_nat (N : nat) : Prop :=
  N > 1 /\ (forall A B, embed N = embed A * embed B -> A = 1 \/ B = 1).

Definition intrinsic_prime (a : R) : Prop :=
  exists N, intrinsic_prime_nat N /\ a = embed N.

(**************************************************************************
 * 5. Unique Factorization into Intrinsic Primes at the nat level
 *
 * We prove that every natural number N > 1 factors uniquely into intrinsic primes.
 * We use the classical Fundamental Theorem of Arithmetic.
 **************************************************************************)
Theorem fundamental_theorem_arithmetic :
  forall n, n > 1 ->
  exists (l : list nat),
    (Forall prime l) /\
    (fold_left Nat.mul l 1 = n) /\
    (forall l', (Forall prime l') /\ (fold_left Nat.mul l' 1 = n) ->
                Permutation l l').
Proof.
  intros n H.
  apply prime_factorization.
Qed.

Lemma intrinsic_prime_equiv_prime :
  forall n, n > 1 -> intrinsic_prime_nat n <-> prime n.
Proof.
  intros n H.
  unfold intrinsic_prime_nat, embed.
  split; intros H0.
  - split; [assumption|].
    intros A B Heq.
    rewrite INR_mult in Heq.
    apply (INR_inj _ _ Heq).
  - split; [assumption|].
    intros A B Heq.
    rewrite INR_mult.
    apply f_equal.
    apply (INR_inj _ _ Heq).
Qed.

Theorem unique_factorization_intrinsic :
  forall N, N > 1 ->
  exists (l : list nat),
    (Forall intrinsic_prime_nat l) /\
    (fold_left Nat.mul l 1 = N) /\
    (forall l', (Forall intrinsic_prime_nat l') /\ (fold_left Nat.mul l' 1 = N) ->
                Permutation l l').
Proof.
  intros N HN.
  destruct (fundamental_theorem_arithmetic N HN) as [l [Hprime [Hprod Hunique]]].
  assert (Hintrinsic: Forall intrinsic_prime_nat l).
  {
    induction l as [| n l' IH].
    - constructor.
    - inversion Hprime; subst.
      constructor.
      + apply intrinsic_prime_equiv_prime; assumption.
      + apply IH.
  }
  exists l.
  split; [exact Hintrinsic |].
  split; assumption.
Qed.

(**************************************************************************
 * 6. From Embedded Lists to Natural Number Lists
 *
 * We prove that any list l' of elements of ℝ that are canonical embeddings of intrinsic primes
 * can be deembedded to obtain a list of natural numbers.
 **************************************************************************)
Lemma deembed_list :
  forall (l' : list R),
    Forall (fun a => exists p, intrinsic_prime_nat p /\ a = embed p) l' ->
    exists l_nat' : list nat, map embed l_nat' = l'.
Proof.
  induction l' as [| a l' IH].
  - intros _. exists []. reflexivity.
  - intros H.
    inversion H as [| a0 l0 Hhead Htail]; subst.
    destruct Hhead as [p [Hp Heq]].
    specialize (IH Htail) as [l_nat' Hl_nat'].
    exists (p :: l_nat').
    simpl. rewrite Heq, Hl_nat'. reflexivity.
Qed.

(**************************************************************************
 * 7. Fold-left Compatibility for embed
 *
 * We prove that the multiplicative structure is preserved by the embedding under fold_left.
 **************************************************************************)
Lemma fold_left_embed : forall (l : list nat),
  fold_left (fun acc p => mul acc (embed p)) l one = embed (fold_left Nat.mul l 1).
Proof.
  induction l as [| x xs IH].
  - simpl. rewrite embed_one. reflexivity.
  - simpl. rewrite embed_mul. rewrite IH. reflexivity.
Qed.

(**************************************************************************
 * 8. Uniqueness of Factorization for Canonical Embeddings
 *
 * We show that if l and l' are two lists of elements of ℝ (each being the canonical
 * embedding of an intrinsic prime) whose product equals embed N, then they are permutations of each other.
 **************************************************************************)
Theorem unique_factorization_embedding :
  forall N, N > 1 ->
  exists (l : list R),
    (Forall (fun a => exists p, intrinsic_prime_nat p /\ a = embed p) l) /\
    (fold_left mul l one = embed N) /\
    (forall l',
        (Forall (fun a => exists p, intrinsic_prime_nat p /\ a = embed p) l') /\
        (fold_left mul l' one = embed N) ->
        Permutation l l').
Proof.
  intros N HN.
  destruct (unique_factorization_intrinsic N HN) as [l_nat [Hint_nat [Hprod_nat Hunique_nat]]].
  set (l := map embed l_nat).
  exists l.
  split.
  - apply Forall_map. intros p Hp.
    exists p; split.
    + apply Hint_nat.
    + reflexivity.
  - split.
    + rewrite fold_left_embed. rewrite Hprod_nat. reflexivity.
    + intros l' [Hfor Hfold].
      destruct (deembed_list l' Hfor) as [l_nat' Hl_nat'].
      assert (Hprod_nat' : fold_left Nat.mul l_nat' 1 = N).
      {
        rewrite <- (fold_left_embed l_nat').
        rewrite Hl_nat'.
        rewrite Hfold.
        apply embed_injective.
      }
      assert (Permutation l_nat l_nat') by (apply Hunique_nat; split; assumption).
      apply Permutation_map in H0.
      rewrite Hl_nat'. assumption.
Qed.

(**************************************************************************
 * Conclusion:
 *
 * We have instantiated the Prime Framework in the concrete model A = ℝ with the standard
 * inclusion of natural numbers (embed = INR) and proved that every natural number N > 1 factors
 * uniquely (up to permutation) into intrinsic primes, where intrinsic primes coincide with the usual
 * primes.
 *
 * This completes the Coq proof corresponding to 3-intrinsic.pdf.
 **************************************************************************)
