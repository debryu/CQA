from loguru import logger
import os
import torch
import wandb
import json
from CQA.utils.vlgcbm_utils import Backbone, BackboneCLIP, ConceptLayer, FinalLayer
from CQA.utils.vlgcbm_utils import get_classes, get_concepts, get_filtered_concepts_and_counts, get_concept_dataloader
from CQA.utils.vlgcbm_utils import get_loss, get_final_layer_dataset, load_concept_and_count, save_filtered_concepts, save_concept_count
from CQA.utils.vlgcbm_utils import train_cbl, train_sparse_final, test_model, per_class_accuracy    

#TODO: fix 12u29nr2

def train(args):
    # Load classes
    classes = get_classes(args.dataset)

    logger.debug(f"VLG classes: {classes}")
    # Load Backbone model
    if args.backbone.startswith("clip_"):
        backbone = BackboneCLIP(
            args.backbone, use_penultimate=args.use_clip_penultimate, device=args.device
        )
    else:
        backbone = Backbone(args.backbone, args.feature_layer, args.device)

    conc_filter = True
    if args.skip_concept_filter:
            conc_filter=False
            #logger.info("Skipping concept filtering")
            #concepts, concept_counts = load_concept_and_count(
            #    os.path.dirname(args.concept_set), filter_file=args.filter_set
            #)
    if False:  # Ugly, but works. Fix it in the future 12u29nr2
        pass  
    else:
        # filter concepts
        logger.info("Filtering concepts")
        raw_concepts = get_concepts(args.concept_set, args.filter_set)
        logger.debug(f"Raw concepts n.{len(raw_concepts)}")

        (
            concepts,
            concept_counts,
            filtered_concepts,
        ) = get_filtered_concepts_and_counts(
            args.dataset,
            raw_concepts,
            preprocess=backbone.preprocess,
            val_split=args.val_split,
            batch_size=args.cbl_batch_size,
            num_workers=args.num_workers,
            confidence_threshold=args.cbl_confidence_threshold,
            label_dir=args.annotation_dir,
            use_allones=args.allones_concept,
            seed=args.seed,
            remove_never_seen=conc_filter
        )
        print(len(concepts))
        # save concept counts
        save_concept_count(concepts, concept_counts, args.save_dir)
        save_filtered_concepts(filtered_concepts, args.save_dir)
    
    with open(os.path.join(args.save_dir, "concepts.txt"), "w") as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write("\n" + concept)

    # setup all dataloaders
    augmented_train_cbl_loader = get_concept_dataloader(
        args.dataset,
        "train",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=True,  # shuffle for training
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=args.crop_to_concept_prob,  # crop to concept
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )
    train_cbl_loader = get_concept_dataloader(
        args.dataset,
        "train",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,  # no shuffle to match order
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,  # no augmentation
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )  # no shuffle to match labels
    val_cbl_loader = get_concept_dataloader(
        args.dataset,
        "val",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,  # no augmentation
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )
    test_cbl_loader = get_concept_dataloader(
        args.dataset,
        "test",
        concepts,
        preprocess=backbone.preprocess,
        val_split=None,  # not needed
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,  # no augmentation
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )

    loss_fn = get_loss(
        args.cbl_loss_type,
        len(concepts),
        len(train_cbl_loader.dataset),
        concept_counts,
        args.cbl_pos_weight,
        args.cbl_auto_weight,
        args.cbl_twoway_tp,
        args.device,
    )

    logger.info("Training CBL")
    cbl = ConceptLayer(
        backbone.output_dim,
        len(concepts),
        num_hidden=args.cbl_hidden_layers,
        device=args.device,
    )
    cbl, backbone = train_cbl(
        backbone,
        cbl,
        augmented_train_cbl_loader,
        val_cbl_loader,
        args.cbl_epochs,
        loss_fn=loss_fn,
        lr=args.cbl_lr,
        weight_decay=args.cbl_weight_decay,
        concepts=concepts,
        tb_writer=None,
        device=args.device,
        finetune=args.cbl_finetune,
        optimizer=args.cbl_optimizer,
        scheduler=args.cbl_scheduler,
        backbone_lr=args.cbl_lr * args.cbl_bb_lr_rate,
        data_parallel=args.data_parallel,
        args = args
    )

    cbl.save_model(args.save_dir)
    if args.cbl_finetune:
        backbone.save_model(args.save_dir)

    ##############################################
    # FINAL layer training
    ##############################################
    (
        train_concept_loader,
        val_concept_loader,
        normalization_layer,
    ) = get_final_layer_dataset(
        backbone,
        cbl,
        train_cbl_loader,
        val_cbl_loader,
        args.save_dir,
        load_dir=None,
        batch_size=args.saga_batch_size,
        device=args.device,
    )

    # Make linear model
    final_layer = FinalLayer(len(concepts), len(classes), device=args.device)

    if args.dense:
        raise NotImplementedError("Sparse final layer training not implemented")
        logger.info(f"Training dense final layer with lr: {args.dense_lr} ...")
        output_proj = train_dense_final(
            final_layer,
            train_concept_loader,
            val_concept_loader,
            args.saga_n_iters,
            args.dense_lr,
            device=args.device,
        )
    else:
        
        logger.info(f"Training sparse final layer ...")
        output_proj = train_sparse_final(
            final_layer,
            train_concept_loader,
            val_concept_loader,
            args.saga_n_iters,
            args.saga_lam,
            step_size=args.saga_step_size,
            device=args.device,
        )

    W_g = output_proj["path"][0]["weight"]
    b_g = output_proj["path"][0]["bias"]
    final_layer.load_state_dict({"weight": W_g, "bias": b_g})
    final_layer.save_model(args.save_dir)

    # save normalized concept features
    #train_concept_features = torch.load(os.path.join(args.save_dir, "train_concept_features.pt"))
    #train_concept_labels = torch.load(os.path.join(args.save_dir, "train_concept_labels.pt"))
    #val_concept_features = torch.load(os.path.join(args.save_dir, "val_concept_features.pt"))
    #val_concept_labels = torch.load(os.path.join(args.save_dir, "val_concept_labels.pt"))
    #from sklearn.svm import LinearSVC
    #from sklearn.metrics import classification_report
    # Train LinearSVC model
    #clf = LinearSVC(C = 1)
    #import numpy as np
    #train_concept_features = train_concept_features.cpu().numpy()
    #train_concept_labels = train_concept_labels.cpu().numpy()
    #clf.fit(train_concept_features, train_concept_labels)
    #all_preds = clf.predict(val_concept_features.cpu().numpy())
    #all_labels = val_concept_labels.cpu().numpy()  # Move to CPU & convert to NumPy
    # Compute classification report
    #report = classification_report(all_labels, all_preds, digits=4)
    #print(report)

    '''
    # Can't do test evaluation here because we don't have annotated the test set
    ##############################################
    #### Test the model on test set ####
    ##############################################
    
    test_accuracy = test_model(
        test_cbl_loader, backbone, cbl, normalization_layer, final_layer, args.device, args=args
    )
    logger.info(f"Test accuracy: {test_accuracy}")

    ##############################################
    # Store training metadata
    ##############################################
    with open(os.path.join(args.save_dir, "vlg-metrics.txt"), "w") as f:
        out_dict = {}
        out_dict["per_class_accuracies"] = per_class_accuracy(
            torch.nn.Sequential(backbone, cbl, normalization_layer, final_layer).to(
                args.device
            ),
            test_cbl_loader,
            classes,
            device=args.device,
        )

        for key in ("lam", "lr", "alpha", "time"):
            out_dict[key] = float(output_proj["path"][0][key])
        out_dict["metrics"] = output_proj["path"][0]["metrics"]
        out_dict["metrics"]["test_accuracy"] = test_accuracy
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict["sparsity"] = {
            "Non-zero weights": nnz,
            "Total weights": total,
            "Percentage non-zero": nnz / total,
        }
        json.dump(out_dict, f, indent=2)
    '''
    return args

def train_last_layer(args):
    ''' #########################################
        ####        TRAIN LAST LAYER         ####
        #########################################
    '''
    pass
