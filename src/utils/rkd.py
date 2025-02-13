import torch
import torch.nn as nn
import torch.nn.functional as F

class RKDLoss(nn.Module):
    def __init__(self, lambda_d=1.0, lambda_a=2.0, temperature=0.5):
        """
        Relational Knowledge Distilaltion Loss with soft attention applied to student embeddings, guided by teacher knowledge.

        Args:
            lambda_d (float): Weight for distance loss.
            lambda_a (float): Weight for angle loss.
            temperature (float): Scaling factor for softmax attention (lower = sharper focus).
        """
        super(RKDLoss, self).__init__()
        self.lambda_d = lambda_d
        self.lambda_a = lambda_a
        self.temperature = temperature

    def compute_pairwise_distances(self, embeddings):
        """
        Compute pairwise distances between embeddings.
        Args:
            embeddings (Tensor): Shape (batch_size, embedding_dim)
        Returns:
            Tensor: Pairwise distance matrix (batch_size, batch_size)
        """
        norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = embeddings / (norm + 1e-6)  
        dist_matrix = torch.cdist(normalized_embeddings, normalized_embeddings, p=2)  
        return dist_matrix

    def soft_attention_student_aggregation(self, teacher_embeddings, student_embeddings):
        """
        Compute attention-weighted student embeddings, guided by the teacher embeddings.

        Args:
            teacher_embeddings (Tensor): (1, num_teacher_points, embedding_dim)
            student_embeddings (Tensor): (1, num_student_points, embedding_dim)

        Returns:
            Tensor: Attention-weighted student embeddings (1, num_teacher_points, embedding_dim)
        """
        # Remove batch dimension
        teacher_embeddings = teacher_embeddings.squeeze(0)  # Shape: (num_teacher, embedding_dim)
        student_embeddings = student_embeddings.squeeze(0)  # Shape: (num_student, embedding_dim)

        # Compute similarity between teacher and student embeddings
        similarity_matrix = torch.matmul(teacher_embeddings, student_embeddings.T)  # Shape: (num_teacher, num_student)

        # Apply temperature-scaled softmax over student embeddings
        attention_weights = F.softmax(similarity_matrix / self.temperature, dim=1)  # Shape: (num_teacher, num_student)

        # Compute attention-weighted sum of student embeddings (students attend to teacher guidance)
        weighted_student_embeddings = torch.matmul(attention_weights, student_embeddings)  # Shape: (num_teacher, embedding_dim)

        # Re-add batch dimension for consistency
        return weighted_student_embeddings.unsqueeze(0)  # Shape: (1, num_teacher, embedding_dim)



    def compute_angle_loss(self, teacher_embeddings, weighted_student_embeddings):
        """
        Compute angle loss between teacher and attention-weighted student embeddings.

        Args:
            teacher_embeddings (Tensor): (batch_size_teacher, embedding_dim)
            weighted_student_embeddings (Tensor): (batch_size_teacher, embedding_dim)

        Returns:
            Tensor: Scalar angle loss
        """
        diff_teacher = teacher_embeddings.unsqueeze(0) - teacher_embeddings.unsqueeze(1)  
        diff_student = weighted_student_embeddings.unsqueeze(0) - weighted_student_embeddings.unsqueeze(1)  

        teacher_angles = F.cosine_similarity(diff_teacher[:, :, None, :], diff_teacher[:, None, :, :], dim=-1)
        student_angles = F.cosine_similarity(diff_student[:, :, None, :], diff_student[:, None, :, :], dim=-1)

        angle_loss = F.mse_loss(student_angles, teacher_angles)
        return angle_loss

    def forward(self, teacher_embeddings, student_embeddings):
        """
        Compute total RKD loss with attention-weighted student embeddings.

        Args:
            teacher_embeddings (Tensor): (batch_size_teacher, embedding_dim)
            student_embeddings (Tensor): (batch_size_student, embedding_dim)

        Returns:
            Tensor: RKD loss value
        """
        # Compute attention-weighted student embedding (student learns from teacher structure)
        # Matches embedding dimension of teacher.
        weighted_student_embeddings = self.soft_attention_student_aggregation(teacher_embeddings, student_embeddings)

        # Compute pairwise distance loss between teacher embeddings & attention-weighted student embeddings
        teacher_distances = self.compute_pairwise_distances(teacher_embeddings)
        student_distances = self.compute_pairwise_distances(weighted_student_embeddings)
        distance_loss = F.mse_loss(student_distances, teacher_distances)

        # Compute angle loss
        angle_loss = self.compute_angle_loss(teacher_embeddings, weighted_student_embeddings)

        # Final RKD loss
        total_loss = self.lambda_d * distance_loss + self.lambda_a * angle_loss
        return total_loss
