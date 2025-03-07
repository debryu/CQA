#DEBUGGING
    train_y = torch.LongTensor(train_targets)
    train_ds = TensorDataset(clip_features, train_y)
    val_y = torch.LongTensor(val_targets)
    val_ds = TensorDataset(val_clip_features,val_y)
    test_y = torch.LongTensor(test_targets)
    test_ds = TensorDataset(test_clip_features,test_y)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=True)
    linear = torch.nn.Linear(clip_features.shape[1],len(classes)).to(args.device)
    
    #linear = torch.nn.Sequential(
    #    torch.nn.Linear(clip_features.shape[1],1000),
    #    torch.nn.ReLU(),
    #    torch.nn.Linear(1000,1000),
    #   torch.nn.ReLU(),
    #    torch.nn.Linear(1000,1000),
    #    torch.nn.ReLU(),
    #    torch.nn.Linear(1000,len(classes))
    #).to(args.device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optim = torch.optim.Adam(params=linear.parameters(), lr = 0.00001)
    from sklearn.svm import LinearSVC
    # Train LinearSVC model
    clf = LinearSVC()
    import numpy as np
    for e in tqdm(range(50)):
        e_l = []
        all_preds = []
        all_labels = []
        for batch in train_loader:
            x,y = batch
            x = x.to('cuda')
            y = y.to('cuda')
            pred_y = linear(x)
            loss = loss_fn(pred_y,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            e_l.append(loss.item())
            # Store predictions and labels
            all_preds.extend(x.cpu().numpy())  # Move to CPU & convert to NumPy
            all_labels.extend(y.cpu().numpy())  # Move to CPU & convert to NumPy
        print(np.mean(e_l))
        break
    clf.fit(all_preds, all_labels)

    

    

    from sklearn.metrics import classification_report
    # Initialize lists to store predictions and ground truth labels
    all_preds = []
    all_labels = []
    for batch in test_loader:
        x,y = batch
        x = x.to('cuda')
        y = y.to('cuda')
        pred_y = torch.argmax(linear(x), dim=1)
        
        # Store predictions and labels
        #all_preds.extend(pred_y.cpu().numpy())  # Move to CPU & convert to NumPy
        # Make predictions
        all_preds.extend(clf.predict(x.cpu().numpy()))
        all_labels.extend(y.cpu().numpy())  # Move to CPU & convert to NumPy
    # Compute classification report
    report = classification_report(all_labels, all_preds, digits=4)
    print(report)
    weights = torch.tensor(clf.coef_)
    print(torch.topk(weights, k=3, largest=True))
    print(torch.topk(weights, k=3, largest=False))
    
    asd



    with torch.no_grad():
        train_y = torch.LongTensor(train_targets)
        indexed_train_ds = IndexedTensorDataset(clip_features, train_y)
        val_y = torch.LongTensor(val_targets)
        val_ds = TensorDataset(val_clip_features,val_y)
        test_y = torch.LongTensor(test_targets)
        test_ds = TensorDataset(test_clip_features,test_y)

    indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds,batch_size=args.saga_batch_size, shuffle=False)

    # Make linear model and zero initialize
    n_concepts_final_layer = clip_features.shape[1]
    logger.debug(f"N. of concepts in the final bottleneck: {n_concepts_final_layer}")
    # add n_concepts to the args
    setattr(args, 'n_concepts_final_layer', n_concepts_final_layer)
    linear = torch.nn.Linear(clip_features.shape[1],len(classes)).to(args.device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()
    
    STEP_SIZE = 0.1
    ALPHA = 0.99
    metadata = {}
    metadata['max_reg'] = {}
    metadata['max_reg']['nongrouped'] = args.lam

    #data = GenericDataset(args.dataset, split='train')
    # Solve the GLM path
    #output_proj = glm_saga(linear, indexed_train_loader, args.glm_step_size, args.n_iters, args.glm_alpha, epsilon=1, k=1,
                      val_loader=val_loader, test_loader=test_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes))
                      #balancing_loss_weight=data.label_weights)
    #W_g = output_proj['path'][0]['weight']
    #b_g = output_proj['path'][0]['bias']